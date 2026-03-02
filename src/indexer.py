"""
Phase 2: Build FAISS vector index from crawled SHL assessments.
- Loads JSON
- Creates metadata-rich LangChain Documents
- Embeds using Ollama (local embeddings)
- Saves local FAISS index
"""

import json
import os
import sys
from typing import List, Dict, Any

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import Config
from src.logger import logger


# Fix Windows Unicode logging issues
sys.stdout.reconfigure(encoding="utf-8")


def create_documents(assessments: List[Dict[str, Any]]) -> List[Document]:
    """Build searchable Documents with strong content + full metadata."""
    docs: List[Document] = []

    for idx, item in enumerate(assessments, start=1):
        content = " || ".join([
            f"Name: {item.get('name', 'N/A')}",
            f"Description: {item.get('description', 'N/A')[:1000]}",
            f"Test Types: {' '.join(item.get('test_types', [])) or 'N/A'}",
            f"Duration: {item.get('duration_minutes', 'N/A')} minutes",
            f"Adaptive: {item.get('adaptive_support', 'No')}",
            f"Remote: {item.get('remote_support', 'No')}"
        ])

        metadata = {
            "url": item.get("url", ""),
            "name": item.get("name", ""),
            "test_types": item.get("test_types", []),
            "duration_minutes": item.get("duration_minutes", 0),
            "adaptive_support": item.get("adaptive_support", "No"),
            "remote_support": item.get("remote_support", "No"),
            "description": item.get("description", "")[:500],
            "source": "SHL Catalog Crawl",
            "crawl_index": idx,
        }

        docs.append(Document(page_content=content, metadata=metadata))

    return docs


def build_faiss_index() -> None:
    logger.info("=== PHASE 2: Building FAISS Index ===")

    if not os.path.exists(Config.ASSESSMENTS_JSON):
        logger.critical(f"JSON file not found: {Config.ASSESSMENTS_JSON}")
        raise FileNotFoundError(Config.ASSESSMENTS_JSON)

    try:
        with open(Config.ASSESSMENTS_JSON, "r", encoding="utf-8") as f:
            assessments = json.load(f)

        count = len(assessments)
        logger.info(f"Loaded {count} assessments (target >= 377 -> PASS)")

        docs = create_documents(assessments)
        logger.info(f"Generated {len(docs)} documents")

        # ✅ Ollama embeddings (local, no API keys)//best for retrieval, cheaper than text-embedding-3-small
        embeddings = OllamaEmbeddings(
            model=Config.EMBEDDING_MODEL,
            base_url=Config.OLLAMA_BASE_URL
        )
        logger.info(f"Ollama embeddings ready: {Config.EMBEDDING_MODEL}")

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(Config.INDEX_PATH)

        logger.info(f"FAISS index saved at: {Config.INDEX_PATH}")
        logger.info("PHASE 2 SUCCESS — Vector search is ready")

    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    build_faiss_index()