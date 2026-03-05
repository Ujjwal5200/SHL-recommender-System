# 🧪 SHL Assessment Recommendation Engine

<p align="center">
  <img src="https://img.shields.io/badge/Version-9.0.0-blue?style=flat-square&logo=version&labelColor=1a1a2e">
  <img src="https://img.shields.io/badge/Python-3.10+-green?style=flat-square&logo=python&labelColor=1a1a2e">
  <img src="https://img.shields.io/badge/License-MIT-orange?style=flat-square&labelColor=1a1a2e">
  <img src="https://img.shields.io/badge/AI-RAG%20Hybrid-purple?style=flat-square&logo=langchain&labelColor=1a1a2e">
</p>

> 🔍 **Intelligent Assessment Matching**: A production-ready RAG-powered system that recommends SHL assessments from natural language job descriptions and queries using hybrid retrieval + LLM reranking.

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Configuration](#️-configuration)
- [📡 API Endpoints](#-api-endpoints)
- [🖥️ Web Interface](#-web-interface)
- [📁 Project Structure](#-project-structure)
- [🧪 Evaluation](#-evaluation)
- [🔧 Troubleshooting](#-troubleshooting)
- [📄 License](#-license)

---

## 🎯 Overview

The **SHL Assessment Recommendation Engine** is an intelligent system designed to match job descriptions and candidate requirements with appropriate SHL assessments. It leverages:

- 🔑 **Keyword Search (Sparse Retrieval)**: TF-weighted matching on assessment names, URLs, test types, and descriptions
- 🧠 **Semantic Search (Dense Retrieval)**: Ollama embeddings + FAISS vector database for meaning-based matching
- 🔗 **Reciprocal Rank Fusion (RRF)**: Combines sparse and dense results for optimal ranking
- 🤖 **LLM Query Understanding**: Multi-stage reranking using Gemini → Ollama fallback chain
- 📊 **Production-Ready**: FastAPI REST API + Streamlit Web UI

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid Retrieval** | Combines keyword and semantic search for comprehensive matching |
| **Multi-Stage Reranking** | Gemini (primary) → Ollama (fallback) for accurate ranking |
| **FAISS Vector Search** | Fast similarity search on 3000+ assessment embeddings |
| **Configurable Weights** | Fine-tune keyword weights, boost factors, and fusion parameters |
| **RESTful API** | FastAPI-based JSON API with health checks |
| **Web UI** | Streamlit-powered interactive interface |
| **Offline Support** | Works without internet using Ollama local models |
| **Evaluation Framework** | Built-in recall@K evaluation on training data |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER QUERY INPUT                                  │
│                   (Job Description / Natural Language)                      │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        📝 QUERY PROCESSING LAYER                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   Keyword Extract   │  │  Semantic Embedding │  │  Stopword Removal   │  │
│  │   (TF-weighted)     │  │  (Ollama/SentenceTr)│  │  (150+ stopwords)   │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      🔄 HYBRID RETRIEVAL LAYER                              │
│                                                                              │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐│
│  │   SPARSE (Keyword Search)  │    │      DENSE (Semantic Search)        ││
│  │  ┌───────────────────────┐  │    │  ┌───────────────────────────────┐  ││
│  │  │ Name Match     × 30   │  │    │  │  Ollama (nomic-embed-text)    │  ││
│  │  │ URL Match      × 20   │  │    │  │  or Sentence-Transformers    │  ││
│  │  │ Test Types    × 25   │  │    │  │  + FAISS Vector Index         │  ││
│  │  │ Description   × 15   │  │    │  │  (768-dim / 384-dim)         │  ││
│  │  └───────────────────────┘  │    │  └───────────────────────────────┘  ││
│  └─────────────────────────────┘    └─────────────────────────────────────┘│
│              │                                      │                       │
│              └──────────────────┬───────────────────┘                       │
│                                 ▼                                            │
│              ┌─────────────────────────────────────────┐                    │
│              │   RECIPROCAL RANK FUSION (RRF, k=60)    │                    │
│              │  Score = Σ 1/(rank + 60) + weighted    │                    │
│              └──────────────────┬─────────────────────┘                    │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       🧠 LLM RERANKING LAYER                                 │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    STAGE CASCADE (Fallback Chain)                    │   │
│  │                                                                       │   │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │   │
│  │   │  Gemini 2.0  │ →  │  Gemini 2.5  │ →  │    Ollama    │          │   │
│  │   │  (Primary)   │    │  (Fallback)  │    │   (Local)    │          │   │
│  │   └──────────────┘    └──────────────┘    └──────────────┘          │   │
│  │                                                                       │   │
│  │   Final Fallback: Basic Keyword Ranking                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          📤 OUTPUT LAYER                                     │
│                                                                              │
│   [                                                                          
│     {                                                                        
│       "name": "Verify/G+ Cognitive Assessment",                             
│       "url": "https://www.shl.com/...",                                     
│       "score": 0.95,                                                        
│       "reason": "Matches cognitive ability, problem solving"               
│     },                                                                       
│     ...                                                                      
│   ]                                                                          
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
python --version

# Optional: Ollama for local embeddings and LLM
# Download from: https://ollama.ai/
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/shl-recommender.git
cd shl-recommender

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install optional LLM dependencies
pip install google-generativeai sentence-transformers

# 5. Set environment variables (optional but recommended)
# Windows
set GEMINI_API_KEY=your_gemini_api_key

# Linux/Mac
export GEMINI_API_KEY=your_gemini_api_key
```

### Running the Application

#### Option 1: FastAPI REST Server

```bash
# Start the API server
python -m uvicorn src.api:app --reload

# Access API docs at: http://localhost:8000/docs
```

#### Option 2: Streamlit Web UI

```bash
# Launch interactive web interface
streamlit run src/streamlit_app.py

# Access at: http://localhost:8501
```

#### Option 3: Direct Python Usage

```bash
# Test the recommender directly
python -c "
from src.recommender import recommend, initialize

initialize()
results = recommend('Java developer assessment for senior role')
for r in results:
    print(f\"- {r['name']}: {r['score']:.2f}\")
"
```

---

## ⚙️ Configuration

All configuration is centralized in `config_v9.py`. Key settings:

```python
# ==================== EMBEDDINGS ====================
EMBED_MODEL = "all-MiniLM-L6-v2"           # Sentence-transformers
OLLAMA_EMBED_MODEL = "nomic-embed-text"    # Ollama embeddings

# ==================== LLM MODELS ====================
GEMINI_MODEL_1 = "gemini-3.1-flash-lite-preview"  # Primary
GEMINI_MODEL_2 = "gemini-2.5-flash-lite"           # Fallback
OLLAMA_LLM_MODEL = "qwen3.5:0.8b"                  # Local fallback

# ==================== RETRIEVAL WEIGHTS ====================
KW_WEIGHT_NAME = 30.0           # Keyword name match
KW_WEIGHT_TEST_TYPES = 25.0    # Test type match
KW_WEIGHT_DESC = 15.0           # Description match
SEMANTIC_TOP_K = 100            # Semantic search candidates
HYBRID_TOP_K = 25               # Final hybrid candidates
RRF_K = 60                      # RRF parameter
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | No | Google Gemini API key for LLM reranking |
| `OLLAMA_MODEL` | No | Ollama model (default: `llama3.2:latest`) |
| `EMBED_MODEL` | No | Embedding model (default: `all-MiniLM-L6-v2`) |

---

## 📡 API Endpoints

### Base Information

```
GET /
```

Returns API metadata and available endpoints.

### Health Check

```
GET /health
```

Returns system health status and version.

### Get Recommendations

```
POST /recommend
```

**Request Body:**

```json
{
  "query": "Java developer assessment for senior role",
  "top_k": 10,
  "use_rerank": true
}
```

**Response:**

```json
{
  "query": "Java developer assessment for senior role",
  "recommended_assessments": [
    {
      "name": "Advanced Java Programming",
      "url": "https://www.shl.com/products/product-catalog/view/...",
      "score": 0.95,
      "reason": "Ranked position 1 - matches ['Technical', 'Coding']"
    }
  ]
}
```

### Example cURL

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query": "Python developer assessment", "top_k": 5}'
```

---

## 🖥️ Web Interface

The Streamlit UI provides an intuitive interface for:

- 📝 Text input for job descriptions
- 📊 Interactive results with scores
- 🎯 Assessment details (URL, test types)
- 🔄 Real-time reranking visualization
- 📈 Evaluation metrics display

```bash
streamlit run src/streamlit_app.py
```

---

## 📁 Project Structure

```
SHL-recommender/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── config_v9.py                 # Central configuration
├── logger.py                    # Logging utilities
├── data/
│   ├── train.csv               # Training data with ground truth
│   ├── test.csv                # Test data
│   ├── shl_individual_tests_*.json  # SHL assessment catalog
│   └── faiss_index/
│       ├── index.faiss         # FAISS vector index
│       └── metadata.json       # Assessment metadata
├── src/
│   ├── __init__.py
│   ├── recommender.py          # Core V9 hybrid recommender
│   ├── api.py                  # FastAPI server
│   ├── indexer.py              # FAISS index builder
│   ├── scraper.py              # SHL catalog scraper
│   └── streamlit_app.py        # Streamlit web UI
└── tests/
    └── test_v9.py              # Unit tests
```

---

## 🧪 Evaluation

### Running Evaluation

```python
from tests.evaluate import compute_recall

# Compute recall metrics on training data
results = compute_recall()
print(f"Mean Recall@10: {results['mean_recall@10']:.2%}")
```

### Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Mean Recall@10 | >50% | Primary metric |
| Mean Recall@5 | >35% | Top-5 accuracy |
| Latency | <2s | P95 response time |

### Test Dataset

- **Training Set**: 10 queries with ground truth assessments
- **Test Set**: 9 unlabeled queries

---

## 🔧 Troubleshooting

### Common Issues

#### 1. FAISS Index Not Found

```
FileNotFoundError: FAISS index not found
```

**Solution**: Rebuild the index:
```bash
python src/indexer.py
```

#### 2. Ollama Not Available

```
Ollama not available
```

**Solution**: Install Ollama or use Gemini API:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Or set Gemini API key
export GEMINI_API_KEY=your_key
```

#### 3. Import Errors

```
ModuleNotFoundError: No module named 'faiss'
```

**Solution**: Install faiss-cpu:
```bash
pip install faiss-cpu
```

#### 4. Gemini API Errors

```
Gemini API error: API key not valid
```

**Solution**: 
- Check your `GEMINI_API_KEY` environment variable
- Ensure billing is enabled on Google Cloud Console

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set in config_v9.py
LOG_LEVEL = logging.DEBUG
```

---

## 📊 Performance Benchmarks

| Version | Approach | Mean Recall@10 | Latency |
|---------|----------|----------------|---------|
| V1 | Basic FAISS + Ollama | ~35% | ~3s |
| V2 | Optimized Keywords | ~42% | ~2s |
| V3 | Enhanced Scoring | ~45% | ~2s |
| V4 | Skill Mappings | ~48% | ~2s |
| V5 | Final Keyword Opt | ~51% | ~1.5s |
| V9 | **RAG Hybrid + LLM** | **>51%** | **<2s** |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [SHL](https://www.shl.com/) for their assessment catalog
- [FAISS](https://github.com/facebookresearch/faiss) by Facebook Research
- [Ollama](https://ollama.ai/) for local LLM capabilities
- [Google Gemini](https://gemini.google.com/) for cloud LLM services

---

<p align="center">
  Made with ❤️ for HR Tech
</p>

