from typing import List, Dict, Any
import json
import os

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from src.config import Config
from src.logger import logger


def load_vectorstore() -> FAISS:
    if not os.path.exists(Config.INDEX_PATH):
        raise FileNotFoundError(f"FAISS index missing: {Config.INDEX_PATH}. Run indexer first.")
    return FAISS.load_local(Config.INDEX_PATH, allow_dangerous_deserialization=True)


def retrieve_candidates(query: str, k: int = 35) -> List[Dict[str, Any]]:
    logger.info(f"Retrieving top-{k} for query: {query[:80]}...")
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    candidates = []
    for doc in docs:
        meta = doc.metadata
        candidates.append({
            "name": meta.get("name", "N/A"),
            "url": meta.get("url", ""),
            "test_types": meta.get("test_types", []),
            "duration_minutes": meta.get("duration_minutes", 0),
            "adaptive_support": meta.get("adaptive_support", "No"),
            "remote_support": meta.get("remote_support", "No"),
            "description": meta.get("description", "N/A")
        })
    logger.info(f"Retrieved {len(candidates)} candidates")
    return candidates


def rerank_with_gemini(
    query: str,
    candidates: List[Dict[str, Any]],
    min_count: int = 5,
    max_count: int = 10,
    preferred_count: int = 7
) -> List[Dict[str, Any]]:
    logger.info("Reranking with Gemini 1.5-flash")

    needs_balance = any(w in query.lower() for w in ["collaborate", "team", "communication", "personality", "leadership", "soft skill", "culture", "fit"]) and \
                    any(w in query.lower() for w in ["java", "python", "sql", "developer", "technical", "programming", "coding"])

    candidates_text = "\n".join([
        f"{i+1}. {c['name']} | {c['url']} | Types: {', '.join(c['test_types'])} | Duration: {c['duration_minutes']} min"
        for i, c in enumerate(candidates)
    ])

    prompt = f"""
You are an expert SHL assessment recommender.

Query: {query}

Candidates:
{candidates_text}

Instructions:
- Choose the MOST RELEVANT {min_count} to {max_count} assessments (target {preferred_count}).
- Balance K (Knowledge & Skills) and P (Personality & Behavior) if query has both technical and soft-skill needs.
- Prefer shorter durations if unspecified.
- Rank by relevance.
- Return ONLY valid JSON array with keys:
  - name (string)
  - url (string)
  - test_types (array)
  - duration_minutes (int)
  - adaptive_support (string)
  - remote_support (string)
  - description (string)

No extra text.
"""

    llm = ChatGoogleGenerativeAI(
        model=Config.GEMINI_MODEL,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=0.1
    )

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()

        if text.startswith("```json"):
            text = text.split("```json")[1].split("```")[0].strip()

        selected = json.loads(text)
        selected = selected[:max_count] if isinstance(selected, list) else []

        if len(selected) < min_count:
            logger.warning(f"Only {len(selected)} returned → fallback")
            selected = candidates[:preferred_count]

        # Simple balance post-process
        if needs_balance:
            k_count = sum("K" in x.get("test_types", []) for x in selected)
            p_count = sum("P" in x.get("test_types", []) for x in selected)
            if k_count == 0 or p_count == 0:
                missing = "P" if k_count == 0 else "K"
                extra = [c for c in candidates if missing in c["test_types"] and c not in selected][:2]
                selected = (selected + extra)[:max_count]

        logger.info(f"Final count: {len(selected)}")
        return selected

    except Exception as e:
        logger.error(f"Gemini failed: {e}")
        return candidates[:preferred_count]


def get_recommendations(query: str) -> List[Dict[str, Any]]:
    candidates = retrieve_candidates(query)
    return rerank_with_gemini(query, candidates)