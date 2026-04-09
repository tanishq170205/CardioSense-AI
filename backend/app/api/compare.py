from fastapi import APIRouter, HTTPException
import json
import os

router = APIRouter()

@router.get("/compare")
async def get_comparison():
    """
    Returns the pre-computed metrics for all models (binary vs multiclass, raw vs fuzzy).
    """
    metrics_path = 'saved_models/metrics.json'
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics not found. Please train models first.")
        
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
