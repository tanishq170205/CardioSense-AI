import shap
import pandas as pd
import numpy as np

def get_shap_values(model, X_instance: pd.DataFrame, model_name: str, background_data: pd.DataFrame = None):
    """
    Computes SHAP values for a single instance for the given model.
    Returns a dictionary of feature names and their corresponding SHAP values.
    """
    # For tree-based models, TreeExplainer is fast
    tree_models = ['Random Forest', 'Decision Tree', 'XGBoost', 'LightGBM']
    
    try:
        if model_name in tree_models:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_instance)
        else:
            # For Linear, SVM, ANN, etc.
            if background_data is None:
                raise ValueError(f"Background data is required for KernelExplainer ({model_name})")
            # We sample the background data to speed up KernelExplainer
            background_sample = shap.sample(background_data, 50)
            explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, 'predict_proba') else model.predict, background_sample)
            shap_values = explainer.shap_values(X_instance)
            
        # Parse output depending on binary vs multiclass
        # For typical binary tree models, shap_values might be a list of arrays (one for each class)
        # or a single array. We want the SHAP values corresponding to the "positive" (disease) class.
        
        if isinstance(shap_values, list):
            # For some scikit-learn models, it's a list where index 1 is the positive class
            target_shap = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif len(shap_values.shape) == 3:
            # For some models (e.g. LightGBM multiclass)
            target_shap = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
        else:
            target_shap = shap_values
            
        # target_shap might be 2D: (1, num_features). Extract the 1D array
        if len(target_shap.shape) > 1:
            target_shap = target_shap[0]
            
        feature_names = X_instance.columns.tolist()
        
        # Combine features with their SHAP impact
        impacts = {}
        for i, feature in enumerate(feature_names):
            val = float(target_shap[i]) if i < len(target_shap) else 0.0
            impacts[feature] = val
            
        return impacts
        
    except Exception as e:
        print(f"Error computing SHAP for {model_name}: {e}")
        return []
