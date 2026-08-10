from typing import Optional
from pydantic import BaseModel
 
 
class OccupationInput(BaseModel):
    """
    One occupation record. Field names must match the merged dataset's
    columns (see data_manager.merge_datasets). Only the identifier and
    feature columns are needed — leakage columns and the target are not
    part of the input.
    """
    occupation_title: str
    soc_code: str
    job_category: str
    education_required: Optional[str] = None
    median_annual_wage_usd: float
    employment_2024: float
 
    class Config:
        extra = "allow"  # allow the remaining cognitive/feature columns through
 
 
class PredictionResponse(BaseModel):
    occupation_title: str
    ai_exposure_level_pred: str
    ai_exposure_level_pred_code: int
 
 
class BatchPredictionRequest(BaseModel):
    records: list[OccupationInput]
 
 
class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
 