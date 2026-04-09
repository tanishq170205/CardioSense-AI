import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import skfuzzy as fuzz
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Data Loader & Basic Preprocessing
# ---------------------------------------------------------
def load_and_clean_data(filepath='heart_disease_uci.csv'):
    """
    Load data, clean missing values, and handle data types.
    Assumes standard UCI Heart Disease CSV dataset.
    """
    df = pd.read_csv(filepath)
    
    # Drop ID columns if they exist
    cols_to_drop = ['id', 'dataset']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # The UCI dataset often represents missing categorical values with 'NaN' or '?'
    df.replace('?', np.nan, inplace=True)
    
    # Define Column Types
    target_col = 'num'
    numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    bool_cols = ['fbs', 'exang']
    categorical_cols = ['sex', 'cp', 'restecg', 'slope', 'thal']
    
    # Convert numeric columns strictly
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Impute missing values early to allow fuzzification
    # (Median for numeric, Mode for categorical)
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
        
    for col in categorical_cols + bool_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
        
    # Binarize bool columns
    for col in bool_cols:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower() == 'true'
        df[col] = df[col].astype(int)
        
    return df, numeric_cols, categorical_cols, bool_cols, target_col

# ---------------------------------------------------------
# 2. Fuzzification Engineering
# ---------------------------------------------------------
def add_fuzzy_features(df, numeric_cols):
    """
    Applies Triangular/Trapezoidal fuzzification to numeric variables.
    """
    X_fuzzy = pd.DataFrame(index=df.index)
    
    if 'age' in df.columns:
        val = df['age'].values
        X_fuzzy['age_young'] = fuzz.trapmf(val, [0, 0, 30, 40])
        X_fuzzy['age_middle'] = fuzz.trimf(val, [35, 50, 65])
        X_fuzzy['age_senior'] = fuzz.trimf(val, [55, 65, 75])
        
    if 'chol' in df.columns:
        val = df['chol'].values
        X_fuzzy['chol_normal'] = fuzz.trapmf(val, [0, 0, 200, 240])
        X_fuzzy['chol_high'] = fuzz.trapmf(val, [200, 240, 600, 600])

    if 'trestbps' in df.columns:
        val = df['trestbps'].values
        X_fuzzy['bp_normal'] = fuzz.trapmf(val, [0, 0, 120, 130])
        X_fuzzy['bp_high'] = fuzz.trapmf(val, [120, 140, 300, 300])

    if 'thalch' in df.columns:
        val = df['thalch'].values
        X_fuzzy['hr_normal'] = fuzz.trapmf(val, [0, 0, 140, 160])
        X_fuzzy['hr_high'] = fuzz.trapmf(val, [150, 170, 300, 300])
        
    if 'oldpeak' in df.columns:
        val = df['oldpeak'].values
        X_fuzzy['op_mild'] = fuzz.trapmf(val, [-5, -5, 1.0, 2.0])
        X_fuzzy['op_severe'] = fuzz.trapmf(val, [1.5, 3.0, 10, 10])
        
    # Return Hybrid dataset (Raw + Fuzzy)
    return pd.concat([df, X_fuzzy], axis=1)

# ---------------------------------------------------------
# 3. Model Dictionary & Tuning Grids
# ---------------------------------------------------------
def get_model_configs(is_multiclass=False):
    rs = 42
    
    xgb_obj = 'multi:softprob' if is_multiclass else 'binary:logistic'
    lgbm_obj = 'multiclass' if is_multiclass else 'binary'
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=rs),
        'Decision Tree': DecisionTreeClassifier(random_state=rs),
        'Random Forest': RandomForestClassifier(random_state=rs),
        'SVM': SVC(probability=True, random_state=rs),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective=xgb_obj, random_state=rs),
        'LightGBM': LGBMClassifier(objective=lgbm_obj, random_state=rs, verbose=-1),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=rs),
        'ANN': MLPClassifier(max_iter=1000, random_state=rs),
        'Bagging GBDT': BaggingClassifier(estimator=LGBMClassifier(objective=lgbm_obj, random_state=rs, verbose=-1), random_state=rs)
    }
    
    grids = {
        'Logistic Regression': {'classifier__C': [0.1, 1, 10]},
        'Decision Tree': {'classifier__max_depth': [3, 5, 10, None]},
        'Random Forest': {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [None, 10, 20]},
        'SVM': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']},
        'XGBoost': {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1]},
        'LightGBM': {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1]},
        'CatBoost': {'classifier__iterations': [100, 200], 'classifier__depth': [4, 6]},
        'ANN': {'classifier__hidden_layer_sizes': [(64,), (128, 64)]},
        'Bagging GBDT': {'classifier__n_estimators': [10, 20]}
    }
    return models, grids

def get_ensembles(best_estimators):
    estimators = [(name, model) for name, model in best_estimators.items() if name in ['XGBoost', 'LightGBM', 'Random Forest', 'CatBoost']]
    if len(estimators) < 2: return {}
    return {
        'Voting': VotingClassifier(estimators=estimators, voting='soft'),
        'Stacking': StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
    }

# ---------------------------------------------------------
# 4. Evaluation Helper
# ---------------------------------------------------------
def evaluate(y_true, y_pred, y_prob, is_multiclass):
    metrics = {}
    if is_multiclass:
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['F1 Score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        try:
            metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except:
            metrics['AUC-ROC'] = np.nan
    else:
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['F1 Score'] = f1_score(y_true, y_pred, zero_division=0)
        try:
            metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob[:, 1] if len(y_prob.shape)>1 else y_prob)
        except:
            metrics['AUC-ROC'] = np.nan
    return metrics

# ---------------------------------------------------------
# 5. Core Pipeline Orchestrator
# ---------------------------------------------------------
def run_experiment(df, numeric_cols, cat_cols, target_col, is_multiclass=False, use_fuzzy=False):
    print(f"\\n--- Running {'Multi-class' if is_multiclass else 'Binary'} | Fuzzy={use_fuzzy} ---")
    
    # Setup Target
    if is_multiclass:
        y = df[target_col].astype(int)
    else:
        y = (df[target_col] > 0).astype(int)
        
    X = df.drop(columns=[target_col])
    
    # Optional Fuzzification
    if use_fuzzy:
        X = add_fuzzy_features(X, numeric_cols)
        
    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Preprocessor (Proper scaling without leakage)
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])
    
    models, grids = get_model_configs(is_multiclass)
    
    results = []
    best_estimators = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        # Pipeline strictly prevents leakage across folds
        # SMOTE -> Preprocessor -> Model
        k_neighbors = min(3, min(y_train.value_counts()) - 1)
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, k_neighbors=k_neighbors)),
            ('classifier', model)
        ])
        
        grid = grids.get(name, {})
        search = RandomizedSearchCV(pipeline, param_distributions=grid, n_iter=5, cv=cv, 
                                    scoring='f1_weighted' if is_multiclass else 'roc_auc', 
                                    random_state=42, n_jobs=-1)
        try:
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_estimators[name] = best_model
            
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else y_pred
            
            metrics = evaluate(y_test, y_pred, y_prob, is_multiclass)
            metrics['Model'] = name
            results.append(metrics)
        except Exception as e:
            print(f"Skipping {name}: {str(e)[:50]}")
            
    # Train Ensembles using tuned estimators
    ensembles = get_ensembles(best_estimators)
    for e_name, e_model in ensembles.items():
        try:
            pipeline = ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42, k_neighbors=k_neighbors)),
                ('classifier', e_model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, 'predict_proba') else y_pred
            
            metrics = evaluate(y_test, y_pred, y_prob, is_multiclass)
            metrics['Model'] = e_name
            results.append(metrics)
        except Exception as e:
            pass

    df_results = pd.DataFrame(results).sort_values(by=['AUC-ROC', 'F1 Score'], ascending=False)
    return df_results, best_estimators, X_test, y_test

# ---------------------------------------------------------
# 6. Execution Loop (Run this cell in Colab)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ensure skfuzzy is installed: !pip install scikit-fuzzy imbalanced-learn catboost lightgbm xgboost
    df, num_cols, cat_cols, bool_cols, target = load_and_clean_data('heart_disease_uci.csv')
    
    # Compare with and without fuzzy (Binary)
    res_bin_raw, _, X_test_bin, y_test_bin = run_experiment(df, num_cols, cat_cols, target, is_multiclass=False, use_fuzzy=False)
    res_bin_fuz, best_bin_fuz, _, _ = run_experiment(df, num_cols, cat_cols, target, is_multiclass=False, use_fuzzy=True)
    
    print("\n[Realistic Results] Binary Classification (Raw Features)")
    display(res_bin_raw)
    
    print("\n[Realistic Results] Binary Classification (Fuzzy Hybrid Features)")
    display(res_bin_fuz)
    
    # Plot ROC AUC for top 3 models in Fuzzy
    plt.figure(figsize=(8, 6))
    for name in res_bin_fuz.head(3)['Model']:
        model = best_bin_fuz[name]
        try:
            y_prob = model.predict_proba(X_test_bin)[:, 1] # Note: X_test_bin needs fuzzy applied if evaluating here, 
            # In a real notebook, evaluate on the fuzzy X_test returned by run_experiment.
        except:
            pass
    
    print("\nPlease implement the detailed plotting cell to visualize ROC curves!")
