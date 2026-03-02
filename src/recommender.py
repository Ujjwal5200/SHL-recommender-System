"""
Core Recommendation Engine for SHL Assessment System
- Loads existing FAISS index (Ollama embeddings)
- Retrieves top candidates via vector similarity
- Reranks with Gemini 1.5-flash (free tier) + balance logic
- Returns 5–10 recommendations in exact Appendix 2 format
"""

import json
import os
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from src.config import Config
from src.logger import logger


def load_vectorstore() -> FAISS:
    """Load pre-built FAISS index from disk."""
    if not os.path.exists(Config.INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {Config.INDEX_PATH}. "
            "Run src/indexer.py first (you already did)."
        )
    return FAISS.load_local(
        Config.INDEX_PATH,
        allow_dangerous_deserialization=True
    )


def retrieve_candidates(query: str, k: int = 35) -> List[Dict[str, Any]]:
    """Retrieve top-k semantically similar assessments from FAISS."""
    logger.info(f"Retrieving top-{k} candidates for query: {query[:80]}...")

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
            "description": meta.get("description", "N/A")[:400]
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
    """
    Use Gemini to select and rank 5–10 most relevant items.
    Enforces balance between K (technical) and P (personality) when query suggests both.
    Returns exact format required by API/Appendix 2.
    """
    logger.info("Starting Gemini reranking (1.5-flash)")

    # Heuristic: does query need both technical + soft skills?
    needs_balance = (
        any(w in query.lower() for w in [
            "collaborate", "team", "communication", "personality", "behavior",
            "leadership", "soft skill", "interpersonal", "culture", "fit"
        ])
        and
        any(w in query.lower() for w in [
            "java", "python", "sql", "developer", "technical", "programming",
            "coding", "engineer", "analyst", "data"
        ])
    )

    # Format candidates for prompt
    candidates_text = "\n".join([
        f"{i+1}. {c['name']} | {c['url']} | Types: {', '.join(c['test_types'])} | "
        f"Duration: {c['duration_minutes']} min | Adaptive: {c['adaptive_support']}"
        for i, c in enumerate(candidates)
    ])

    prompt_text = f"""
You are an expert at recommending SHL individual assessments.

User query:
{query}

Available candidates:
{candidates_text}

Instructions:
- Select EXACTLY between {min_count} and {max_count} most relevant assessments (aim for ~{preferred_count}).
- If query mentions both technical skills AND collaboration/soft skills, balance Knowledge & Skills (K) and Personality & Behavior (P) types.
- Prefer shorter durations when duration not specified in query.
- Rank from most to least relevant.
- Return ONLY a valid JSON array of objects. Each object MUST have exactly these keys:
  - name (string)
  - url (string)
  - test_types (array of strings)
  - duration_minutes (integer)
  - adaptive_support (string: "Yes" or "No")
  - remote_support (string: "Yes" or "No")
  - description (string – short version, max 300 chars)

No extra text, explanations, or markdown outside the JSON array.
"""

    llm = ChatGoogleGenerativeAI(
        model=Config.GEMINI_MODEL,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=0.1,
        max_output_tokens=1500
    )

    try:
        response = llm.invoke(prompt_text)
        text = response.content.strip()

        # Clean Gemini's common wrappers
        if text.startswith("```json"):
            text = text.split("```json")[1].split("```")[0].strip()
        elif text.startswith("```"):
            text = text.split("```")[1].strip()

        selected = json.loads(text)

        if not isinstance(selected, list):
            raise ValueError("Gemini did not return a list")

        selected = selected[:max_count]

        if len(selected) < min_count:
            logger.warning(f"Gemini returned only {len(selected)} items → using fallback")
            selected = candidates[:preferred_count]

        # Post-process balance if needed
        if needs_balance:
            k_count = sum(1 for x in selected if "K" in x.get("test_types", []))
            p_count = sum(1 for x in selected if "P" in x.get("test_types", []))
            if k_count == 0 or p_count == 0:
                logger.info("Applying balance correction")
                missing_type = "P" if k_count == 0 else "K"
                extra = [
                    c for c in candidates
                    if missing_type in c["test_types"] and c not in selected
                ][:2]
                selected.extend(extra)
                selected = selected[:max_count]

        logger.info(f"Final recommendation count: {len(selected)}")
        return selected

    except Exception as e:
        logger.error(f"Gemini reranking failed: {str(e)}", exc_info=True)
        logger.warning("Falling back to top retrieved candidates")
        return candidates[:preferred_count]


def get_recommendations(query: str) -> List[Dict[str, Any]]:
    """
    One-call public function: retrieve → rerank → return recommendations.
    Used by API and frontend.
    """
    try:
        candidates = retrieve_candidates(query)
        recommendations = rerank_with_gemini(query, candidates)
        return recommendations
    except Exception as e:
        logger.critical(f"Full recommendation pipeline failed: {e}", exc_info=True)
        return []