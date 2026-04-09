# CardioSense AI
CardioSense AI is a Python-based machine learning project for accurate classification of heart disease using the UCI Heart Disease dataset. It leverages advanced ML models, fuzzy logic, and ensemble techniques to perform both binary and multi-class classification.

Installation

Clone the repository and install the required dependencies:

git clone https://github.com/your-username/CardioSense-AI.git
cd CardioSense-AI
pip install -r requirements.txt
Usage
from train import run_experiment, load_and_clean_data

# Load dataset
df, num_cols, cat_cols, bool_cols, target = load_and_clean_data('heart_disease_uci.csv')

# Run binary classification
results, models, X_test, y_test = run_experiment(df, target, is_multiclass=False, use_fuzzy=True)

# View results
print(results.head())
Features
Binary classification (Disease vs No Disease)
Multi-class classification (Severity 0–4)
Fuzzy logic-based feature engineering
Advanced models:
Logistic Regression, SVM, Decision Tree
Random Forest, Extra Trees
XGBoost, LightGBM, CatBoost
ANN (MLP Classifier)
Ensemble learning:
Voting Classifier
Stacking Classifier
SMOTE for class imbalance
Stratified K-Fold Cross Validation
Hyperparameter tuning (RandomizedSearchCV)
Evaluation metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC
Contributing

Contributions are welcome.
For major changes, please open an issue first to discuss what you would like to improve.

Notes

This project is intended for educational and research purposes only and should not be used for medical diagnosis.