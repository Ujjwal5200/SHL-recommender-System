import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DATA_DIR = "data"
    INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
    ASSESSMENTS_JSON = os.path.join(DATA_DIR, "shl_individual_tests.json")
    EMBEDDING_MODEL = "models/embedding-001"
    LLM_MODEL = "gemini-1.5-flash"          # or gemini-1.5-pro if quota allows