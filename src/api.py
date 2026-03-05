# FastAPI API - V9 Production Ready using config_v9.py
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

from src.recommender import recommend, initialize
from config_v9 import logger, LOG_LEVEL
import logging

# Configure logging
logging.basicConfig(level=LOG_LEVEL)

app = FastAPI(
    title="SHL Recommender API V9",
    description="RAG-powered assessment recommendation",
    version="9.0.0"
)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    use_rerank: Optional[bool] = True


@app.on_event("startup")
async def startup():
    logger.info("Starting V9 API...")
    try:
        initialize()
        logger.info("Recommender initialized")
    except Exception as e:
        logger.error(f"Init error: {e}")


@app.get("/")
def root():
    return {
        "name": "SHL Assessment Recommender API",
        "version": "9.0.0",
        "endpoints": ["/health", "/recommend"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "9.0.0",
        "gemini_available": os.environ.get("GEMINI_API_KEY") is not None
    }


@app.post("/recommend")
def get_recommendations(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    top_k = min(request.top_k or 10, 20)
    
    try:
        results = recommend(query, top_k=top_k, use_rerank=request.use_rerank)
        return {"recommended_assessments": results, "query": query}
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
