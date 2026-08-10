from fastapi import FastAPI
 
from app.api import router
 
app = FastAPI(
    title="Will AI Take My Job? — Exposure Classifier",
    description="Predicts AI exposure level (Low/Medium/High) for a given occupation.",
    version="1.0.0",
)
 
app.include_router(router)
 
 
@app.get("/")
def root():
    return {"message": "Will AI Take My Job? API is running. See /docs for usage."}
 