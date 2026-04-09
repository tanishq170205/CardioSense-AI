from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_models(task='binary'):
    # task: 'binary' or 'multi'
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'ExtraTrees': ExtraTreesClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss' if task=='binary' else 'mlogloss'),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
        'BaggedGBDT': BaggingClassifier(estimator=XGBClassifier(random_state=42, eval_metric='logloss' if task=='binary' else 'mlogloss'), n_estimators=10, random_state=42),
        'GNN_Proxy_MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    }
    
    # Define hyperparameter search spaces for RandomizedSearchCV
    param_grids = {
        'LogisticRegression': {'C': [0.01, 0.1, 1, 10]},
        'KNN': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
        'ExtraTrees': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
        'XGBoost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
        'LightGBM': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'num_leaves': [31, 50]},
        'CatBoost': {'iterations': [50, 100], 'learning_rate': [0.01, 0.1], 'depth': [4, 6]},
        'BaggedGBDT': {'n_estimators': [5, 10]},
        'GNN_Proxy_MLP': {'alpha': [0.0001, 0.001], 'learning_rate_init': [0.001, 0.01]}
    }
    
    return models, param_grids

def get_ensembles(best_estimators):
    """
    Constructs Voting and Stacking classifiers from best estimators
    """
    estimators = [(name, model) for name, model in best_estimators.items() if name in ['RandomForest', 'SVM', 'XGBoost']]
    if len(estimators) < 2:
        return {}, {}
        
    ensembles = {
        'Voting_Soft': VotingClassifier(estimators=estimators, voting='soft'),
        'Stacking': StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    }
    
    param_grids = {
        'Voting_Soft': {}, # usually no hyperparams for voting
        'Stacking': {}
    }
    return ensembles, param_grids
