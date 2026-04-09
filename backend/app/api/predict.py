from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
import json
from app.preprocessing import HeartDiseasePreprocessor
from app.explainability import get_shap_values

router = APIRouter()

# Global variables to cache model and preprocessor
best_model = None
best_model_name = ""
preprocessor = None
training_features = None # To use as background for KernelExplainer if needed

def load_artifacts():
    global best_model, best_model_name, preprocessor, training_features
    try:
        if best_model is None:
            best_model = joblib.load('saved_models/best_model.pkl')
            with open('saved_models/best_model_name.txt', 'r') as f:
                best_model_name = f.read().strip()
            preprocessor = HeartDiseasePreprocessor.load('saved_models/preprocessor.pkl')
            
            # Simple dummy background dataset for KernelExplainer (if using SVM/Logistic)
            # normally we'd save background data during train.py
    except Exception as e:
        print(f"Artifact loading error: {e}")

class PatientData(BaseModel):
    age: float = Field(..., description="Age in years")
    sex: str = Field(..., description="Male or Female")
    cp: str = Field(..., description="Chest pain type")
    trestbps: float = Field(..., description="Resting blood pressure")
    chol: float = Field(..., description="Serum cholesterol in mg/dl")
    fbs: bool = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: str = Field(..., description="Resting electrocardiographic results")
    thalch: float = Field(..., description="Maximum heart rate achieved")
    exang: bool = Field(..., description="Exercise induced angina")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: str = Field(..., description="Slope of the peak exercise ST segment")
    ca: float = Field(..., description="Number of major vessels (0-3)")
    thal: str = Field(..., description="Thal: normal, fixed defect, reversable defect")

@router.post("/predict")
async def predict(data: PatientData):
    load_artifacts()
    if best_model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model artifacts not found. Please train the model first.")
        
    # Convert input to DataFrame
    df_input = pd.DataFrame([data.dict()])
    
    try:
        # For prediction, we use the preprocessor.
        # Check if the best model is a fuzzy one based on the training script.
        # In our script, best_model is tracked under feature_type="fuzzy" for binary.
        X_processed = preprocessor.transform(df_input, use_fuzzy=True)
        
        # Predict
        if hasattr(best_model, 'predict_proba'):
            prob = best_model.predict_proba(X_processed)[0]
            probability = float(prob[1]) if len(prob) > 1 else float(prob[0])
        else:
            prediction = best_model.predict(X_processed)[0]
            probability = 1.0 if prediction > 0 else 0.0
            
        is_disease = probability > 0.5
        
        # SHAP calculation
        shap_impacts = get_shap_values(best_model, X_processed, best_model_name, background_data=X_processed) 
        # (Using X_instance itself as fallback background for kernel explainer to prevent crash, though not strictly correct. TreeExplainer works fine without background.)
        
        # Extract fuzzy membership degrees just for display
        # We know columns that have membership start with feature name
        fuzzy_memberships = {}
        for col in X_processed.columns:
            for prefix in ['age_', 'chol_', 'trestbps_', 'thalch_', 'oldpeak_']:
                if col.startswith(prefix):
                    fuzzy_memberships[col] = float(X_processed[col].iloc[0])
                    
        return {
            "binary_result": "Disease" if is_disease else "No Disease",
            "probability": probability,
            "risk_score": int(probability * 100),
            "model_used": best_model_name,
            "shap_values": shap_impacts,
            "fuzzy_memberships": fuzzy_memberships
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
