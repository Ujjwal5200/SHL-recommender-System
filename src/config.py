import os
from dotenv import load_dotenv

load_dotenv()

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
    ASSESSMENTS_JSON = os.path.join(DATA_DIR, "shl_individual_tests_20260302_1257.json")

    # Ollama (only used if you want local rerank fallback)
    OLLAMA_BASE_URL = "http://localhost:11434"
    EMBEDDING_MODEL = "nomic-embed-text:latest"   # already built index with this
    OLLAMA_MODEL = "gemma3:1b"              # for reranking candidates (fallback if Gemini API fails) 

    # Gemini (for reranking & structured output)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = "gemini-2.5-flash"             # fast + free tier generous
