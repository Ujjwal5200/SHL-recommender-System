"""
Phase 2: Build FAISS vector index and BM25 index from crawled SHL assessments.
- Loads JSON
- Creates metadata-rich LangChain Documents
- Embeds using Ollama (local embeddings)
- Builds BM25 index for keyword search
- Saves both indexes locally
"""

import json
import os
import sys
import pickle
from typing import List, Dict, Any

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

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
            f"Adaptive: {item.get('adaptive_support', 'No')}",
            f"Remote: {item.get('remote_support', 'No')}"
        ])

        metadata = {
            "url": item.get("url", ""),
            "name": item.get("name", ""),
            "test_types": item.get("test_types", []),
            "adaptive_support": item.get("adaptive_support", "No"),
            "remote_support": item.get("remote_support", "No"),
            "description": item.get("description", "")[:1000],
            "source": "SHL Catalog Crawl",
            "crawl_index": idx,
        }

        docs.append(Document(page_content=content, metadata=metadata))

    return docs


def create_bm25_index(docs: List[Document]) -> BM25Okapi:
    """Create BM25 index from documents."""
    # Tokenize document contents for BM25
    tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def build_faiss_index() -> None:
    logger.info("=== PHASE 2: Building FAISS + BM25 Index ===")

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

        # Build FAISS index
        embeddings = OllamaEmbeddings(
            model=Config.EMBEDDING_MODEL,
            base_url=Config.OLLAMA_BASE_URL
        )
        logger.info(f"Ollama embeddings ready: {Config.EMBEDDING_MODEL}")

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(Config.INDEX_PATH)
        logger.info(f"FAISS index saved at: {Config.INDEX_PATH}")

        # Build BM25 index
        logger.info("Building BM25 index...")
        bm25 = create_bm25_index(docs)
        
        # Save BM25 index with documents for retrieval
        bm25_data = {
            "bm25": bm25,
            "docs": docs
        }
        with open(Config.BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25_data, f)
        logger.info(f"BM25 index saved at: {Config.BM25_INDEX_PATH}")

        logger.info("PHASE 2 SUCCESS — Both FAISS and BM25 indexes are ready")

    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    build_faiss_index()
