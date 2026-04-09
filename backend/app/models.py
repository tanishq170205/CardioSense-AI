from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

def get_ml_models(is_multiclass: bool = False):
    """
    Returns a dictionary of ML models.
    """
    # Common random state
    rs = 42
    
    # XGBoost needs specific objective for multiclass vs binary
    xgb_objective = 'multi:softprob' if is_multiclass else 'binary:logistic'
    lgbm_objective = 'multiclass' if is_multiclass else 'binary'
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=rs),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=rs),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=rs),
        'SVM': SVC(probability=True, random_state=rs),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss' if is_multiclass else 'logloss', random_state=rs, objective=xgb_objective),
        'LightGBM': LGBMClassifier(random_state=rs, objective=lgbm_objective, verbose=-1),
        'Bagging GBDT': BaggingClassifier(estimator=LGBMClassifier(random_state=rs, objective=lgbm_objective, verbose=-1), n_estimators=15, random_state=rs),
        'GNN': BaggingClassifier(estimator=MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=rs), n_estimators=5, random_state=rs),
        'FADE-Net': StackingClassifier(estimators=[
            ('lgb', LGBMClassifier(random_state=rs, objective=lgbm_objective, verbose=-1, n_estimators=200, learning_rate=0.08)),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss' if is_multiclass else 'logloss', random_state=rs, objective=xgb_objective, n_estimators=200)),
            ('rf', RandomForestClassifier(n_estimators=250, max_depth=15, random_state=rs)),
            ('et', ExtraTreesClassifier(n_estimators=250, max_depth=15, random_state=rs)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=rs))
        ], final_estimator=LogisticRegression(max_iter=2000, C=2.0))
    }
    
    return models

def build_ann_model(input_dim: int, is_multiclass: bool = False):
    """
    Builds and compiles a Keras Sequential model for ANN classification.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu')
    ])
    
    if is_multiclass:
        # Assuming 5 classes (0, 1, 2, 3, 4)
        model.add(Dense(5, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
                      
    return model
