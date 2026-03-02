import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATA_DIR = "data"
    INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
    ASSESSMENTS_JSON = os.path.join(DATA_DIR, "shl_individual_tests_20260302_1257.json")

    # Ollama (only used if you want local rerank fallback)
    OLLAMA_BASE_URL = "http://localhost:11434"
    EMBEDDING_MODEL = "nomic-embed-text:latest"   # already built index with this

    # Gemini (for reranking & structured output)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = "gemini-2.5-flash"             # fast + free tier generous