import numpy as np
import pandas as pd
import skfuzzy as fuzz

def fuzzify_age(age_col):
    """
    Fuzzify age into Young, Middle-aged, Senior, Old.
    Returns a DataFrame with the membership degrees.
    """
    age_range = np.arange(0, 101, 1)
    # Define membership functions
    young_mf = fuzz.trapmf(age_range, [0, 0, 35, 45])
    middle_mf = fuzz.trimf(age_range, [35, 45, 55])
    senior_mf = fuzz.trimf(age_range, [45, 55, 65])
    old_mf = fuzz.trapmf(age_range, [55, 65, 100, 100])
    
    young_mem = fuzz.interp_membership(age_range, young_mf, age_col)
    middle_mem = fuzz.interp_membership(age_range, middle_mf, age_col)
    senior_mem = fuzz.interp_membership(age_range, senior_mf, age_col)
    old_mem = fuzz.interp_membership(age_range, old_mf, age_col)
    
    return pd.DataFrame({
        'age_young': young_mem,
        'age_middle': middle_mem,
        'age_senior': senior_mem,
        'age_old': old_mem
    }, index=age_col.index)

def fuzzify_cholesterol(chol_col):
    """
    Fuzzify cholesterol into Low, Normal, High, Very High.
    """
    chol_range = np.arange(0, 1001, 1)
    low_mf = fuzz.trapmf(chol_range, [0, 0, 150, 200])
    normal_mf = fuzz.trimf(chol_range, [150, 200, 240])
    high_mf = fuzz.trimf(chol_range, [200, 240, 300])
    very_high_mf = fuzz.trapmf(chol_range, [240, 300, 1000, 1000])
    
    return pd.DataFrame({
        'chol_low': fuzz.interp_membership(chol_range, low_mf, chol_col),
        'chol_normal': fuzz.interp_membership(chol_range, normal_mf, chol_col),
        'chol_high': fuzz.interp_membership(chol_range, high_mf, chol_col),
        'chol_very_high': fuzz.interp_membership(chol_range, very_high_mf, chol_col)
    }, index=chol_col.index)

def fuzzify_trestbps(bps_col):
    """
    Fuzzify resting blood pressure into Normal, Elevated, High, Crisis.
    """
    bps_range = np.arange(0, 301, 1)
    normal_mf = fuzz.trapmf(bps_range, [0, 0, 120, 130])
    elev_mf = fuzz.trimf(bps_range, [120, 130, 140])
    high_mf = fuzz.trimf(bps_range, [130, 140, 160])
    crisis_mf = fuzz.trapmf(bps_range, [140, 160, 300, 300])
    
    return pd.DataFrame({
        'trestbps_normal': fuzz.interp_membership(bps_range, normal_mf, bps_col),
        'trestbps_elevated': fuzz.interp_membership(bps_range, elev_mf, bps_col),
        'trestbps_high': fuzz.interp_membership(bps_range, high_mf, bps_col),
        'trestbps_crisis': fuzz.interp_membership(bps_range, crisis_mf, bps_col)
    }, index=bps_col.index)

def fuzzify_thalch(thalch_col):
    """
    Fuzzify maximum heart rate achieved into Low, Normal, High.
    """
    thal_range = np.arange(0, 301, 1)
    low_mf = fuzz.trapmf(thal_range, [0, 0, 100, 130])
    normal_mf = fuzz.trimf(thal_range, [100, 140, 180])
    high_mf = fuzz.trapmf(thal_range, [140, 180, 300, 300])
    
    return pd.DataFrame({
        'thalch_low': fuzz.interp_membership(thal_range, low_mf, thalch_col),
        'thalch_normal': fuzz.interp_membership(thal_range, normal_mf, thalch_col),
        'thalch_high': fuzz.interp_membership(thal_range, high_mf, thalch_col)
    }, index=thalch_col.index)

def fuzzify_oldpeak(oldpeak_col):
    """
    Fuzzify ST depression (oldpeak).
    """
    # oldpeak can be negative occasionally in some datasets, but usually 0+
    op_range = np.arange(-5, 20, 0.1)
    none_mf = fuzz.trapmf(op_range, [-5, -5, 0.5, 1.0])
    mild_mf = fuzz.trimf(op_range, [0.5, 1.0, 2.0])
    mod_mf = fuzz.trimf(op_range, [1.0, 2.0, 3.0])
    sev_mf = fuzz.trapmf(op_range, [2.0, 3.0, 20, 20])
    
    return pd.DataFrame({
        'oldpeak_none': fuzz.interp_membership(op_range, none_mf, oldpeak_col),
        'oldpeak_mild': fuzz.interp_membership(op_range, mild_mf, oldpeak_col),
        'oldpeak_moderate': fuzz.interp_membership(op_range, mod_mf, oldpeak_col),
        'oldpeak_severe': fuzz.interp_membership(op_range, sev_mf, oldpeak_col)
    }, index=oldpeak_col.index)

def apply_fuzzification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies fuzzification to a given dataframe that has standard heart disease columns.
    Returns ONLY the fuzzy features concatenated together.
    """
    fuzzy_dfs = []
    
    if 'age' in df.columns:
        fuzzy_dfs.append(fuzzify_age(df['age']))
    if 'chol' in df.columns:
        fuzzy_dfs.append(fuzzify_cholesterol(df['chol']))
    if 'trestbps' in df.columns:
        fuzzy_dfs.append(fuzzify_trestbps(df['trestbps']))
    if 'thalch' in df.columns:
        fuzzy_dfs.append(fuzzify_thalch(df['thalch']))
    if 'oldpeak' in df.columns:
        fuzzy_dfs.append(fuzzify_oldpeak(df['oldpeak']))
        
    if not fuzzy_dfs:
        return pd.DataFrame(index=df.index)
        
    return pd.concat(fuzzy_dfs, axis=1)

def get_fuzzy_plot_data():
    """
    Returns X, Y arrays for all defined fuzzy membership functions so they can be plotted on the frontend.
    """
    data = {}
    
    # Age
    age_x = np.arange(0, 101, 1)
    data['age'] = {
        'x': age_x.tolist(),
        'lines': [
            {'name': 'Young', 'y': fuzz.trapmf(age_x, [0, 0, 35, 45]).tolist()},
            {'name': 'Middle-aged', 'y': fuzz.trimf(age_x, [35, 45, 55]).tolist()},
            {'name': 'Senior', 'y': fuzz.trimf(age_x, [45, 55, 65]).tolist()},
            {'name': 'Old', 'y': fuzz.trapmf(age_x, [55, 65, 100, 100]).tolist()}
        ]
    }
    
    # Chol
    chol_x = np.arange(0, 501, 5)
    data['chol'] = {
        'x': chol_x.tolist(),
        'lines': [
            {'name': 'Low', 'y': fuzz.trapmf(chol_x, [0, 0, 150, 200]).tolist()},
            {'name': 'Normal', 'y': fuzz.trimf(chol_x, [150, 200, 240]).tolist()},
            {'name': 'High', 'y': fuzz.trimf(chol_x, [200, 240, 300]).tolist()},
            {'name': 'Very High', 'y': fuzz.trapmf(chol_x, [240, 300, 1000, 1000]).tolist()}
        ]
    }
    
    # Trestbps
    bps_x = np.arange(80, 221, 2)
    data['trestbps'] = {
        'x': bps_x.tolist(),
        'lines': [
            {'name': 'Normal', 'y': fuzz.trapmf(bps_x, [0, 0, 120, 130]).tolist()},
            {'name': 'Elevated', 'y': fuzz.trimf(bps_x, [120, 130, 140]).tolist()},
            {'name': 'High', 'y': fuzz.trimf(bps_x, [130, 140, 160]).tolist()},
            {'name': 'Crisis', 'y': fuzz.trapmf(bps_x, [140, 160, 300, 300]).tolist()}
        ]
    }
    
    # Thalch
    thal_x = np.arange(60, 221, 2)
    data['thalch'] = {
        'x': thal_x.tolist(),
        'lines': [
            {'name': 'Low', 'y': fuzz.trapmf(thal_x, [0, 0, 100, 130]).tolist()},
            {'name': 'Normal', 'y': fuzz.trimf(thal_x, [100, 140, 180]).tolist()},
            {'name': 'High', 'y': fuzz.trapmf(thal_x, [140, 180, 300, 300]).tolist()}
        ]
    }
    
    # Oldpeak
    op_x = np.arange(0, 6.1, 0.1)
    data['oldpeak'] = {
        'x': np.round(op_x, 1).tolist(),
        'lines': [
            {'name': 'None', 'y': fuzz.trapmf(op_x, [-5, -5, 0.5, 1.0]).tolist()},
            {'name': 'Mild', 'y': fuzz.trimf(op_x, [0.5, 1.0, 2.0]).tolist()},
            {'name': 'Moderate', 'y': fuzz.trimf(op_x, [1.0, 2.0, 3.0]).tolist()},
            {'name': 'Severe', 'y': fuzz.trapmf(op_x, [2.0, 3.0, 20, 20]).tolist()}
        ]
    }

    return data
