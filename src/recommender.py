"""
Core Recommendation Engine for SHL Assessment System
- Loads existing FAISS index (built with Ollama/nomic-embed-text)
- Retrieves top candidates
- Reranks with Gemini 1.5-flash + balance logic
- Returns 5–10 items in Appendix 2 format
"""

import json
import os
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from src.config import Config
from src.logger import logger


def load_vectorstore() -> FAISS:
    """Load pre-built FAISS index — MUST provide the same embeddings used to create it."""
    if not os.path.exists(Config.INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {Config.INDEX_PATH}. Run indexer first."
        )

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url=Config.OLLAMA_BASE_URL
    )

    logger.info("Loading FAISS index with Ollama embeddings (nomic-embed-text)")

    return FAISS.load_local(
        folder_path=Config.INDEX_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )


def retrieve_candidates(query: str, k: int = 35) -> List[Dict[str, Any]]:
    logger.info(f"Retrieving top-{k} for: {query[:80]}...")
    
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
    logger.info("Reranking with Gemini 1.5-flash")

    needs_balance = any(w in query.lower() for w in [
        "collaborate", "team", "communication", "personality", "leadership",
        "soft skill", "interpersonal", "culture", "fit"
    ]) and any(w in query.lower() for w in [
        "java", "python", "sql", "developer", "technical", "programming", "coding"
    ])

    candidates_text = "\n".join([
        f"{i+1}. {c['name']} | {c['url']} | Types: {', '.join(c['test_types'])} | "
        f"Adaptive: {c['adaptive_support']}"
        for i, c in enumerate(candidates)
    ])

    prompt_text = f"""
You are an expert SHL assessment recommender.

Query: {query}

Candidates:
{candidates_text}

Instructions:
- Select EXACTLY between 5 and 10 most relevant assessments (aim for ~7).
- If query has both technical AND soft/collaboration needs, balance K and P types.
- Rank by relevance to query.
- Return ONLY valid JSON array of objects with keys:
  - name (string)
  - url (string)
  - test_types (array of strings)
  - duration_minutes (int)   # don't include
  - adaptive_support (string)
  - remote_support (string)
  - description (string)

No extra text.
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

        if text.startswith("```json"):
            text = text.split("```json")[1].split("```")[0].strip()

        selected = json.loads(text)
        selected = selected[:max_count] if isinstance(selected, list) else []

        if len(selected) < min_count:
            logger.warning(f"Only {len(selected)} items → fallback")
            selected = candidates[:preferred_count]

        if needs_balance:
            k_count = sum("K" in x.get("test_types", []) for x in selected)
            p_count = sum("P" in x.get("test_types", []) for x in selected)
            if k_count == 0 or p_count == 0:
                logger.info("Applying balance correction")
                missing = "P" if k_count == 0 else "K"
                extra = [c for c in candidates if missing in c["test_types"] and c not in selected][:2]
                selected.extend(extra)
                selected = selected[:max_count]

        logger.info(f"Final count: {len(selected)}")
        return selected

    except Exception as e:
        logger.error(f"Gemini failed: {e}")
        return candidates[:preferred_count]


def get_recommendations(query: str) -> List[Dict[str, Any]]:
    try:
        candidates = retrieve_candidates(query)
        return rerank_with_gemini(query, candidates)
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        return []