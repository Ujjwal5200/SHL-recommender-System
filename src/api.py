from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Use optimized recommender (best performing - 51% recall)
from src.recommender_optimized import get_recommendations
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
        
        formatted = []
        for r in recs:
            formatted.append({
                "name": r.get("name", ""),
                "url": r.get("url", ""),
                "test_types": r.get("test_types", []),
                "duration_minutes": r.get("duration_minutes", 0),
                "adaptive_support": r.get("adaptive_support", "No"),
                "remote_support": r.get("remote_support", "No"),
                "description": r.get("description", "")
            })
        return {"recommended_assessments": formatted}
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(500, str(e))
