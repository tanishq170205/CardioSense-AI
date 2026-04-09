import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

def build_preprocessed_data(X, y, use_smote=True):
    """
    Normalizes numeric features, One-Hot Encodes categorical features, 
    and applies SMOTE for class imbalance.
    """
    # Identify column types
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    # 1. OneHotEncode categorical
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat_enc = pd.DataFrame(encoder.fit_transform(X[cat_cols]))
        X_cat_enc.columns = encoder.get_feature_names_out(cat_cols)
        X_cat_enc.index = X.index
    else:
        X_cat_enc = pd.DataFrame(index=X.index)
        
    # 2. Scale numeric
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X_num_scaled = pd.DataFrame(scaler.fit_transform(X[num_cols]), columns=num_cols, index=X.index)
    else:
        X_num_scaled = pd.DataFrame(index=X.index)
        
    # Combine
    X_processed = pd.concat([X_num_scaled, X_cat_enc], axis=1)
    
    # Apply SMOTE
    if use_smote:
        # SMOTE needs at least a few samples of the minority class
        sm = SMOTE(k_neighbors=min(3, min(y.value_counts()) - 1), random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X_processed, y)
        return X_resampled, y_resampled
    
    return X_processed, y
