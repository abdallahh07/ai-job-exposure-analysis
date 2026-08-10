import pandas as pd
from fastapi import APIRouter, HTTPException
 
from app.schemas import BatchPredictionRequest, BatchPredictionResponse, PredictionResponse
from predict import predict
 
router = APIRouter()
 
 
@router.post("/predict", response_model=BatchPredictionResponse)
def predict_exposure(request: BatchPredictionRequest):
    if not request.records:
        raise HTTPException(status_code=400, detail="records cannot be empty")
 
    try:
        input_df = pd.DataFrame([r.dict() for r in request.records])
        result = predict(input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
 
    predictions = [
        PredictionResponse(
            occupation_title=row["occupation_title"],
            ai_exposure_level_pred=row["ai_exposure_level_pred"],
            ai_exposure_level_pred_code=int(row["ai_exposure_level_pred_code"]),
        )
        for _, row in result.iterrows()
    ]
 
    return BatchPredictionResponse(predictions=predictions)
 
 
@router.get("/health")
def health_check():
    return {"status": "ok"}