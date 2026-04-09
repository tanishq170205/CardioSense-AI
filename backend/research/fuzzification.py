import pandas as pd
import skfuzzy as fuzz

def fuzzify_dataset(X):
    """
    Applies Triangular and Trapezoidal membership functions to specific clinically relevant features.
    Returns the purely fuzzy feature set, and the hybrid feature set.
    """
    X_fuzzy = pd.DataFrame(index=X.index)
    
    # 1. Age
    if 'age' in X.columns:
        val = X['age'].values
        X_fuzzy['age_young'] = fuzz.trapmf(val, [0, 0, 30, 40])
        X_fuzzy['age_middle'] = fuzz.trimf(val, [35, 50, 65])
        X_fuzzy['age_senior'] = fuzz.trimf(val, [55, 65, 75])
        X_fuzzy['age_old'] = fuzz.trapmf(val, [65, 75, 120, 120])
    
    # 2. Cholesterol
    if 'chol' in X.columns:
        val = X['chol'].values
        X_fuzzy['chol_low'] = fuzz.trapmf(val, [0, 0, 150, 200])
        X_fuzzy['chol_normal'] = fuzz.trimf(val, [150, 200, 250])
        X_fuzzy['chol_high'] = fuzz.trimf(val, [200, 260, 320])
        X_fuzzy['chol_veryhigh'] = fuzz.trapmf(val, [280, 400, 1000, 1000])

    # 3. Resting BP
    if 'trestbps' in X.columns:
        val = X['trestbps'].values
        X_fuzzy['bp_normal'] = fuzz.trapmf(val, [0, 0, 110, 120])
        X_fuzzy['bp_elevated'] = fuzz.trimf(val, [115, 125, 135])
        X_fuzzy['bp_high'] = fuzz.trimf(val, [130, 145, 160])
        X_fuzzy['bp_crisis'] = fuzz.trapmf(val, [150, 180, 300, 300])

    # 4. Max Heart Rate
    if 'thalch' in X.columns:
        val = X['thalch'].values
        X_fuzzy['hr_low'] = fuzz.trapmf(val, [0, 0, 100, 130])
        X_fuzzy['hr_normal'] = fuzz.trimf(val, [110, 140, 170])
        X_fuzzy['hr_high'] = fuzz.trapmf(val, [150, 180, 250, 250])

    # 5. Oldpeak
    if 'oldpeak' in X.columns:
        val = X['oldpeak'].values
        X_fuzzy['op_none'] = fuzz.trapmf(val, [-5, -5, 0, 0.5])
        X_fuzzy['op_mild'] = fuzz.trimf(val, [0, 1.0, 2.0])
        X_fuzzy['op_moderate'] = fuzz.trimf(val, [1.5, 3.0, 4.5])
        X_fuzzy['op_severe'] = fuzz.trapmf(val, [3.0, 5.0, 20, 20])
        
    X_hybrid = pd.concat([X.copy(), X_fuzzy], axis=1)
    
    return X_fuzzy, X_hybrid
