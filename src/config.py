import os
from dotenv import load_dotenv

load_dotenv()

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    """Configuration settings for SHL Assessment Recommender."""

    # Data paths
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
    BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
    ASSESSMENTS_JSON = os.path.join(DATA_DIR, "shl_individual_tests_20260302_1257.json")

    # Ollama settings (for embeddings and LLM fallback)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL = "nomic-embed-text:latest"
    OLLAMA_MODEL = "qwen2.5:latest"
    GEMMA_MODEL = "gemma3:1b"  # Fallback when Gemini fails

    # Google Gemini settings (primary for reranking & query expansion)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = "gemini-2.5-flash"

    # LLM choice for reranking: "gemini", "ollama", or "gemma"
    RERANK_LLM = "gemini"

    # Hybrid Search Settings
    VECTOR_TOP_K = 50
    BM25_TOP_K = 50
    RERANK_TOP_K = 20
    FINAL_TOP_K = 10

    # RRF (Reciprocal Rank Fusion) weights
    VECTOR_WEIGHT = 0.7
    BM25_WEIGHT = 0.3

    # Reranker model
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    USE_RERANKER = True

    # Query Expansion Settings
    USE_QUERY_EXPANSION = True
    NUM_QUERY_VARIATIONS = 3
