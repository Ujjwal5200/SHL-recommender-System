# SHL Recommender Configuration
# All configurable parameters in one place

from pathlib import Path
import logging
import os

# ==================== PATHS ====================
# Use parent.parent because config_v9.py is in src/ folder
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR         = ROOT / "data"
CATALOG_GLOB     = DATA_DIR / "shl_individual_tests_*.json"
FAISS_DIR        = DATA_DIR / "faiss_index"
TRAIN_CSV        = DATA_DIR / "train.csv"
TEST_PRED_CSV    = DATA_DIR / "test.csv"

# ==================== SCRAPING ====================
SCRAPE_DELAY_MIN = 1.4
SCRAPE_DELAY_MAX = 3.8
MAX_PAGES        = 40

# ==================== EMBEDDINGS ====================
# Primary embedding model (sentence-transformers)
EMBED_MODEL      = "all-MiniLM-L6-v2"

# Ollama fallback for embeddings
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# ==================== LLM SETTINGS ====================
# Gemini models (fallback order: 1 -> 2 -> Ollama)
GEMINI_MODEL_1   = "gemini-3.1-flash-lite-preview"   # Primary
GEMINI_MODEL_2   = "gemini-2.5-flash-lite"   # Fallback 1
RERANK_MODEL_1   = "gemini-3.1-flash-lite-preview"   # Primary rerank
RERANK_MODEL_2   = "gemini-2.5-flash-lite"   # Fallback rerank

# Ollama (final fallback)
OLLAMA_LLM_MODEL = "qwen3.5:0.8b"

# Legacy (kept for compatibility)
GEMINI_MODEL     = GEMINI_MODEL_2
RERANK_MODEL     = RERANK_MODEL_2

# ==================== RETRIEVAL SETTINGS ====================
# Hybrid retrieval weights
KW_WEIGHT_NAME = 30.0
KW_WEIGHT_TEST_TYPES = 25.0
KW_WEIGHT_DESC = 15.0
KW_MULTIPLIER = 7.0

# Semantic search settings
SEMANTIC_TOP_K = 100
HYBRID_TOP_K = 25
RRF_K = 60

# Keyword boost weights
KEYWORD_BOOST_EXACT = 100.0
KEYWORD_BOOST_PREFIX = 60.0
KEYWORD_BOOST_PARTIAL = 30.0

# ==================== GEMINI RERANKING ====================
RERANK_CANDIDATES = 20  # Number of candidates to send to Gemini
RERANK_MODEL = "gemini-2.5-flash-lite"  # Model for reranking (fallback order: 1 -> 2 -> Ollama)

# ==================== SKILL MARKERS ====================
SOFT_MARKERS = {
    "collaborate", "team", "communication", "leadership", "people management",
    "interpersonal", "cultural fit", "personality", "behavior", "emotional",
    "coach", "mentor", "relationship", "stakeholder", "adaptability", "culture"
}

HARD_MARKERS = {
    "java", "python", "sql", "developer", "coding", "programming",
    "analyst", "engineer", "technical", "excel", "data", "tableau", "script"
}

# ==================== STOPWORDS ====================
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
    'they', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why',
    'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once', 'if',
    'else', 'when', 'while', 'about', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'between', 'under', 'again', 'further',
    'experience', 'years', 'work', 'job', 'role', 'position', 'looking',
    'hire', 'hiring', 'candidates', 'candidate', 'required', 'need', 'skills',
    'based', 'below', 'recommend', 'assessment', 'description', 'purpose',
    'responsibilities', 'duties', 'qualifications', 'company', 'team',
    'my', 'our', 'want', 'find', 'me', 'us', 'get', 'min', 'mins',
    'hour', 'hours', 'less', 'more', 'than', 'able', 'will', 'join',
    'must', 'should', 'would', 'could', 'like', 'make'
}

# ==================== LOGGING ====================
LOG_LEVEL = logging.INFO
LOG_FILE = ROOT / "logs" / "app.log"

logging.basicConfig(
    level=LOG_LEVEL, 
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
)
logger = logging.getLogger("shl-reco")
