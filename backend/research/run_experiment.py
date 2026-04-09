import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import joblib

from data_loader import load_data
from fuzzification import fuzzify_dataset
from models import get_models, get_ensembles
from evaluator import evaluate_model

import warnings
warnings.filterwarnings('ignore')

REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs('saved_models_research', exist_ok=True)

def run_experiment_pipeline():
    results = []
    best_pipelines_cache = {}
    
    tasks = ['binary', 'multi']
    data_modes = ['Raw', 'Fuzzy', 'Hybrid']
    
    for task in tasks:
        print(f"\\n{'='*40}\\nStarting Task: {task.upper()}\\n{'='*40}")
        X_all, y = load_data('../../heart_disease_uci.csv', task=task)
        
        # Train test split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_all, y, test_size=0.2, stratify=y, random_state=42)
        
        # Fuzzify
        X_train_fuz, X_train_hyb = fuzzify_dataset(X_train_raw)
        X_test_fuz, X_test_hyb = fuzzify_dataset(X_test_raw)
        
        datasets = {
            'Raw': (X_train_raw, X_test_raw),
            'Fuzzy': (X_train_fuz, X_test_fuz),
            'Hybrid': (X_train_hyb, X_test_hyb)
        }
        
        models, param_grids = get_models(task)
        
        for mode in data_modes:
            X_tr, X_te = datasets[mode]
            
            print(f"  --> Dataset Mode: {mode}")
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Setup dynamic tracking for ensembling
            mode_best_estimators = {}
            
            for m_name, model in models.items():
                smote = SMOTE(k_neighbors=min(3, min(y_train.value_counts()) - 1), random_state=42)
                
                # Imblearn pipeline: StandardScaler -> SMOTE -> Model
                pipeline = make_pipeline(StandardScaler(), smote, model)
                
                # Map param grids
                grid = {f"{model.__class__.__name__.lower()}__{k}": v for k, v in param_grids[m_name].items()}
                
                search = RandomizedSearchCV(
                    pipeline, 
                    param_distributions=grid, 
                    n_iter=5, 
                    scoring='roc_auc' if task=='binary' else 'f1_weighted', 
                    cv=cv, 
                    random_state=42,
                    n_jobs=-1
                )
                
                try:
                    search.fit(X_tr, y_train)
                    best_pipe = search.best_estimator_
                    
                    # Evaluate
                    metrics = evaluate_model(best_pipe, X_te, y_test, task=task)
                    
                    row = {'Task': task, 'Mode': mode, 'Model': m_name}
                    row.update(metrics)
                    results.append(row)
                    
                    mode_best_estimators[m_name] = best_pipe.steps[-1][1]
                    
                    joblib.dump(best_pipe, f"saved_models_research/{task}_{mode}_{m_name}.pkl")
                    print(f"      [{m_name}] AUC: {metrics['AUC']:.4f} | F1: {metrics['F1 Score']:.4f}")
                except Exception as e:
                    print(f"      [{m_name}] Failed: {str(e)[:50]}")
            
            # Ensembles
            ensembles, ens_grids = get_ensembles(mode_best_estimators)
            for e_name, e_model in ensembles.items():
                smote = SMOTE(k_neighbors=min(3, min(y_train.value_counts()) - 1), random_state=42)
                pipe = make_pipeline(StandardScaler(), smote, e_model)
                try:
                    pipe.fit(X_tr, y_train)
                    metrics = evaluate_model(pipe, X_te, y_test, task=task)
                    row = {'Task': task, 'Mode': mode, 'Model': e_name}
                    row.update(metrics)
                    results.append(row)
                    print(f"      [{e_name}] AUC: {metrics['AUC']:.4f} | F1: {metrics['F1 Score']:.4f}")
                except Exception as e:
                    pass

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{REPORTS_DIR}/metrics_comparison.csv", index=False)
    
    generate_plots(df_res)
    generate_insights(df_res)

def generate_plots(df):
    for task in df['Task'].unique():
        df_task = df[df['Task'] == task]
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_task, x='Model', y='AUC', hue='Mode')
        plt.title(f'AUC Comparison across Models and Features ({task.upper()})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{REPORTS_DIR}/{task}_auc_comparison.png")
        plt.close()

def generate_insights(df):
    best_overall = df.sort_values(by=['AUC', 'F1 Score'], ascending=False).iloc[0]
    
    # Calculate mode average AUC
    mode_avg = df.groupby('Mode')['AUC'].mean().sort_values(ascending=False)
    
    insight = f"""# Experimental Insight Report

## Best Overall Model
- **Model**: {best_overall['Model']}
- **Task**: {best_overall['Task']}
- **Feature Set**: {best_overall['Mode']}
- **AUC**: {best_overall['AUC']:.4f}
- **F1 Score**: {best_overall['F1 Score']:.4f}

## Does Fuzzification Help?
Average AUC across all models by Feature Set:
- **Raw**: {mode_avg.get('Raw', 0.0):.4f}
- **Fuzzy**: {mode_avg.get('Fuzzy', 0.0):.4f}
- **Hybrid**: {mode_avg.get('Hybrid', 0.0):.4f}

*Analysis*: 
Based on empirical evaluation, using {mode_avg.index[0]} features provided the highest average performance. 
"""
    with open(f"{REPORTS_DIR}/ExperimentalInsight.md", "w") as f:
        f.write(insight)

if __name__ == '__main__':
    run_experiment_pipeline()
    print("Experiment completed. Check reports/ folder.")
