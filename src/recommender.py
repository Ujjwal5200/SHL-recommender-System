# src/recommender.py
"""
Correct SHL Assessment Recommender

Key insight:
- SHL tests are language-agnostic
- Rerank by role dimensions, not programming language
"""

import os
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from src.config import Config
from src.logger import logger


# ---------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------

def load_vectorstore() -> FAISS:
    embeddings = OllamaEmbeddings(
        model=Config.EMBEDDING_MODEL,
        base_url=Config.OLLAMA_BASE_URL
    )
    return FAISS.load_local(
        Config.INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


# ---------------------------------------------------------------------
# Role dimension extraction (THIS MATTERS)
# ---------------------------------------------------------------------

def extract_dimensions(query: str) -> Dict[str, int]:
    q = query.lower()

    return {
        "cognitive": int(any(w in q for w in ["problem", "logic", "analytical", "reasoning"])),
        "technical": int(any(w in q for w in ["developer", "engineer", "programmer"])),
        "people": int(any(w in q for w in ["team", "collaborate", "communication"])),
        "leadership": int(any(w in q for w in ["lead", "manager", "management"])),
    }


# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------

def retrieve_candidates(query: str, k: int = 50) -> List[Dict[str, Any]]:
    retriever = load_vectorstore().as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    candidates = []
    for d in docs:
        m = d.metadata
        candidates.append({
            "name": m.get("name", ""),
            "url": m.get("url", ""),
            "test_types": m.get("test_types", []),
            "duration_minutes": m.get("duration_minutes", 0),
            "adaptive_support": m.get("adaptive_support", "No"),
            "remote_support": m.get("remote_support", "No"),
            "description": m.get("description", "")[:300],
        })

    return candidates


# ---------------------------------------------------------------------
# Dimension-based reranking (REAL DIFFERENTIATION)
# ---------------------------------------------------------------------

def rerank(candidates: List[Dict[str, Any]], dims: Dict[str, int]) -> List[Dict[str, Any]]:
    scored = []

    for c in candidates:
        score = 0
        types = set(c["test_types"])

        if dims["cognitive"]:
            score += 3
        if dims["technical"] and "K" in types:
            score += 2
        if dims["people"] and "P" in types:
            score += 2
        if dims["leadership"] and "P" in types:
            score += 3

        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def get_recommendations(query: str) -> List[Dict[str, Any]]:
    logger.info("Starting recommendation pipeline")

    dims = extract_dimensions(query)
    logger.info(f"Extracted dimensions: {dims}")

    candidates = retrieve_candidates(query)
    reranked = rerank(candidates, dims)

    final = reranked[:7]
    logger.info(f"Final recommendation count: {len(final)}")
    return final