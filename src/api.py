# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.recommender import get_recommendations
from src.logger import logger

app = FastAPI(title="SHL Assessment Recommendation API")

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health_check():
    """Health endpoint for monitoring."""
    return {"status": "healthy"}

@app.post("/recommend")
def recommend_assessments(request: QueryRequest):
    """Main recommendation endpoint – returns 5–10 items in Appendix 2 format."""
    logger.info(f"API request received: {request.query[:100]}...")
    
    try:
        recommendations = get_recommendations(request.query)
        
        if not recommendations:
            raise ValueError("No recommendations generated")
        
        # Ensure format matches Appendix 2 exactly
        formatted = [
            {
                "name": r["name"],
                "url": r["url"],
                "test_types": r["test_types"],
                "adaptive_support": r["adaptive_support"],
                "remote_support": r["remote_support"],
                "description": r["description"]
            }
            for r in recommendations
        ]
        
        return {"recommended_assessments": formatted}
    
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))