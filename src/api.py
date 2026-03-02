from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.recommender import get_recommendations
from src.logger import logger

app = FastAPI(title="SHL Assessment Recommender API")

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/recommend")
def recommend(request: QueryRequest):
    logger.info(f"API request: {request.query[:100]}...")
    try:
        recs = get_recommendations(request.query)
        if len(recs) < 5:
            logger.warning("Low count – returning fallback")
        
        formatted = [
            {
                "name": r["name"],
                "url": r["url"],
                "test_types": r["test_types"],
                "duration_minutes": r["duration_minutes"],
                "adaptive_support": r["adaptive_support"],
                "remote_support": r["remote_support"],
                "description": r["description"][:500]
            }
            for r in recs
        ]
        return {"recommended_assessments": formatted}
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(500, str(e))