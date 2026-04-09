import pandas as pd
import numpy as np
import os

def load_data(file_path='../../heart_disease_uci.csv', task='binary'):
    """
    Loads UCI Heart Disease CSV, handles missing values, and formats the target.
    Task can be 'binary' (0 vs 1) or 'multi' (0-4).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Handle missing values (e.g. dropping id, dataset column if exists)
    drop_cols = ['id', 'dataset']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # Impute missing numeric with median, categorical with mode
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('num', errors='ignore')
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    
    for c in numeric_cols:
        df[c].fillna(df[c].median(), inplace=True)
    for c in cat_cols:
        df[c].fillna(df[c].mode()[0], inplace=True)
        
    # Format target
    if task == 'binary':
        y = (df['num'] > 0).astype(int)
    else:
        y = df['num'].astype(int)
        
    X = df.drop(columns=['num'])
    X = pd.get_dummies(X, drop_first=True)
    return X, y
