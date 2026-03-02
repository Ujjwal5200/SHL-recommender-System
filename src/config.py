# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DATA_DIR = "data"
    INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
    ASSESSMENTS_JSON = os.path.join(DATA_DIR, "shl_individual_tests_20260302_1257.json")  # ← your file
  #  EMBEDDING_MODEL = "gemini-embedding-001"  # best for retrieval, cheaper than text-embedding-3-small
  #  LLM_MODEL = "gemini-1.5-flash"          # cheap & fast for reranking

    # Ollama settings
    OLLAMA_BASE_URL = "http://localhost:11434"           # default Ollama server
    EMBEDDING_MODEL = "nomic-embed-text:latest"          # or just "nomic-embed-text"

    # For later reranking (you can use Ollama LLM too)
    LLM_MODEL = "llama3.2:latest"                            # or mistral, gemma2, etc. — choose one you have pulled