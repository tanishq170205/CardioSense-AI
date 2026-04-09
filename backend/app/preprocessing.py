import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
from .fuzzy import apply_fuzzification

NUMERIC_COLS = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
CATEGORICAL_COLS = ['sex', 'cp', 'restecg', 'slope', 'thal']
BOOL_COLS = ['fbs', 'exang']
TARGET_COL = 'num'

class HeartDiseasePreprocessor:
    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.is_fitted = False
        
    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()
        
        # Replace empty strings with NaN if any
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        # Convert numeric columns to float
        for col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Impute missing values
        df[NUMERIC_COLS] = self.numeric_imputer.fit_transform(df[NUMERIC_COLS])
        df[CATEGORICAL_COLS] = self.categorical_imputer.fit_transform(df[CATEGORICAL_COLS])
        
        # Boolean handling
        for col in BOOL_COLS:
            df[col] = df[col].astype(str).str.upper() == 'TRUE'
            df[col] = df[col].astype(int)
            
        # Extract target
        y_multi = df[TARGET_COL].astype(int)
        y_binary = (y_multi > 0).astype(int)
        
        # Select features before encoding (useful for fuzzification)
        raw_numeric_df = df[NUMERIC_COLS].copy()
        
        # One-hot encoding for categoricals
        encoded_cats = self.ohe.fit_transform(df[CATEGORICAL_COLS])
        encoded_cols = self.ohe.get_feature_names_out(CATEGORICAL_COLS)
        df_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols, index=df.index)
        
        # Scale numeric features
        scaled_numeric = self.scaler.fit_transform(df[NUMERIC_COLS])
        scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=NUMERIC_COLS, index=df.index)
        
        # Combine
        X_raw = pd.concat([scaled_numeric_df, df_encoded, df[BOOL_COLS]], axis=1)
        
        # Generate Fuzzified features
        fuzzy_features = apply_fuzzification(raw_numeric_df)
        
        # X_fuzzy is strictly features WITH fuzzy columns instead of numeric
        X_fuzzy = pd.concat([fuzzy_features, df_encoded, df[BOOL_COLS]], axis=1)
        
        self.is_fitted = True
        self.feature_names_raw = X_raw.columns.tolist()
        self.feature_names_fuzzy = X_fuzzy.columns.tolist()
        
        return X_raw, X_fuzzy, y_binary, y_multi
        
    def transform(self, df: pd.DataFrame, use_fuzzy: bool = False):
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted yet.")
            
        df = df.copy()
        
        for col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col not in df.columns:
                df[col] = 0
                
        df[NUMERIC_COLS] = self.numeric_imputer.transform(df[NUMERIC_COLS])
        
        # Handle cases where categorical might be missing in incoming request
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                df[col] = self.categorical_imputer.statistics_[CATEGORICAL_COLS.index(col)]
                
        df[CATEGORICAL_COLS] = self.categorical_imputer.transform(df[CATEGORICAL_COLS])
        
        for col in BOOL_COLS:
            if col in df.columns:
                df[col] = str(df.iloc[0][col]).upper() == 'TRUE' if not isinstance(df.iloc[0][col], bool) else df[col]
                df[col] = df[col].astype(int)
            else:
                df[col] = 0
        
        raw_numeric_df = df[NUMERIC_COLS].copy()
        
        # Make sure dummies match exact columns from training
        encoded_cats = self.ohe.transform(df[CATEGORICAL_COLS])
        encoded_cols = self.ohe.get_feature_names_out(CATEGORICAL_COLS)
        df_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols, index=df.index)
        
        # Align with training features
        target_cols = self.feature_names_fuzzy if use_fuzzy else self.feature_names_raw
        
        scaled_numeric = self.scaler.transform(df[NUMERIC_COLS])
        scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=NUMERIC_COLS, index=df.index)
        
        if use_fuzzy:
            fuzzy_features = apply_fuzzification(raw_numeric_df)
            X = pd.concat([fuzzy_features, df_encoded, df[BOOL_COLS]], axis=1)
        else:
            X = pd.concat([scaled_numeric_df, df_encoded, df[BOOL_COLS]], axis=1)
            
        # Realign columns (fill missing with 0)
        for col in target_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[target_cols]
        
        return X

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str):
        return joblib.load(filepath)
