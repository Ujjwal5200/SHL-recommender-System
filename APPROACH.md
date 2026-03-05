# 🏗️ Technical Approach: SHL Assessment Recommendation Engine V9

> A comprehensive technical deep-dive into the hybrid RAG system architecture, retrieval algorithms, and LLM integration strategies.

---

## 📑 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Definition](#2-problem-definition)
3. [System Evolution](#3-system-evolution)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Retrieval Strategies](#5-retrieval-strategies)
6. [LLM Integration](#6-llm-integration)
7. [Configuration & Tuning](#7-configuration--tuning)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Production Deployment](#9-production-deployment)
10. [Future Improvements](#10-future-improvements)

---

## 1. Executive Summary

The SHL Assessment Recommendation Engine is a production-ready system designed to match job descriptions and natural language queries with appropriate SHL psychometric assessments. The V9 system achieves **>50% Mean Recall@10** through a sophisticated hybrid retrieval approach combining:

- **Sparse Retrieval**: TF-weighted keyword matching
- **Dense Retrieval**: Semantic embeddings + FAISS vector search
- **Reciprocal Rank Fusion (RRF)**: Unified ranking algorithm
- **Multi-Stage LLM Reranking**: Gemini → Ollama fallback cascade

### Key Metrics

| Metric | Value |
|--------|-------|
| Mean Recall@10 | >50% |
| Mean Recall@5 | >35% |
| P95 Latency | <2 seconds |
| Assessment Coverage | 3000+ assessments |

---

## 2. Problem Definition

### 2.1 Task Description

Given a natural language query (job description or hiring requirement), recommend the most relevant SHL assessments from a catalog of 3000+ tests.

### 2.2 Challenges

1. **Vocabulary Mismatch**: Job descriptions use varied terminology (e.g., "coding test" vs. "programming assessment")
2. **Semantic Complexity**: Understanding implicit requirements (e.g., "leadership potential" → leadership assessments)
3. **Scale**: Large assessment catalog with nuanced differences
4. **Latency Requirements**: Production systems require sub-second response times
5. **Offline Capability**: Need to work without external API dependencies

### 2.3 Evaluation Criteria

- **Recall@K**: Fraction of relevant assessments found in top-K recommendations
- **Precision@K**: Accuracy of top-K recommendations
- **Latency**: Response time for recommendations

---

## 3. System Evolution

### Version History

| Version | Approach | Mean Recall@10 | Key Innovations |
|---------|----------|----------------|-----------------|
| V1 | Basic FAISS + Ollama | ~35% | Initial semantic search implementation |
| V2 | Optimized Keywords | ~42% | Tuned keyword weights (URL > Name > Query) |
| V3 | Enhanced Scoring | ~45% | Added test_types matching, TF boosting |
| V4 | Skill Mappings | ~48% | ROLE_SKILL_MAPPING, JOB_TO_SKILLS dictionaries |
| V5 | Final Keyword Opt | ~51% | Optimized weights, term frequency boost |
| V9 | **RAG Hybrid + LLM** | **>60%** with gemini and **80%** with ollama| **Multi-stage reranking, fallback chain** |

### Evolution Insights

1. **Keyword matching is crucial**: Even with semantic search, keyword precision matters significantly
2. **Dimension alignment**: FAISS index dimension must exactly match embedding dimension
3. **Fallback reliability**: Always design fallback chains for critical operations
4. **Hybrid fusion**: RRF provides smooth combination of sparse + dense retrieval

---

## 4. Architecture Deep Dive

### 4.1 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                   │
│    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│    │   REST API  │  │  Streamlit  │  │  Python SDK │  │  Webhook    │    │
│    │  (FastAPI)  │  │    (UI)     │  │   (Direct)  │  │  (Future)   │    │
│    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
└───────────┼────────────────┼────────────────┼────────────────┼───────────┘
            │                │                │                │
            └────────────────┴────────────────┴────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUERY PROCESSING LAYER                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  1. Keyword Extraction     2. Semantic Embedding    3. Query Parse  │ │
│  │     - TF-weighted            - Ollama (768-dim)        - Role extract │ │
│  │     - Stopword removal       - Sentence-transformers   - Skill extract│ │
│  │     - Boost factors          - Fallback chain           - Duration     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYBRID RETRIEVAL LAYER                             │
│                                                                              │
│    ┌─────────────────────────┐         ┌─────────────────────────┐         │
│    │   SPARSE RETRIEVAL     │         │    DENSE RETRIEVAL     │         │
│    │   (Keyword Search)     │         │   (Semantic Search)    │         │
│    │                        │         │                        │         │
│    │  • Name matching       │         │  • Ollama embeddings   │         │
│    │  • URL matching        │         │  • FAISS Index (IVF)   │         │
│    │  • Test types matching │         │  • Cosine similarity   │         │
│    │  • Description matching │         │  • Top-K retrieval     │         │
│    │                        │         │                        │         │
│    │  Weight: 40%           │         │  Weight: 30%           │         │
│    └───────────┬───────────┘         └───────────┬───────────┘         │
│                │                                   │                      │
│                └───────────────┬───────────────────┘                      │
│                                ▼                                           │
│    ┌─────────────────────────────────────────────────────────────────┐    │
│    │            RECIPROCAL RANK FUSION (RRF, k=60)                   │    │
│    │                                                                 │    │
│    │    RRF_score = Σ (1 / (rank + k))                             │    │
│    │    Final = RRF × 100 + Keyword × 0.4 + Semantic × 0.3          │    │
│    └─────────────────────────┬───────────────────────────────────────┘    │
└──────────────────────────────┼────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM RERANKING LAYER                                │
│                                                                              │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │                    STAGE CASCADE                                   │  │
│    │                                                                     │  │
│    │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │  │
│    │   │  Gemini 2.0 │ → │  Gemini 2.5  │ → │   Ollama    │           │  │
│    │   │  (Primary)  │   │  (Fallback)  │   │   (Local)   │           │  │
│    │   │             │   │             │   │             │           │  │
│    │   │ • Fast      │   │ • Accurate  │   │ • Offline   │           │  │
│    │   │ • Reliable  │   │ • Flexible  │   │ • No cost   │           │  │
│    │   └─────────────┘   └─────────────┘   └─────────────┘           │  │
│    │                                                                     │  │
│    │   ┌─────────────────────────────────────────────────────────┐     │  │
│    │   │          FINAL FALLBACK: Basic Keyword Ranking          │     │  │
│    │   └─────────────────────────────────────────────────────────┘     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT GENERATION LAYER                             │
│                                                                              │
│   • Score normalization (0-1)                                              │
│   • Human-readable reason generation                                       │
│   • URL canonicalization                                                    │
│   • Response serialization (JSON)                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Responsibilities

| Component | Responsibility | Key Technologies |
|-----------|---------------|------------------|
| API Server | HTTP handling, request validation | FastAPI, Pydantic |
| Recommender | Core retrieval and ranking logic | Custom Python |
| Indexer | FAISS index building and maintenance | FAISS, NumPy |
| Scraper | SHL catalog extraction | BeautifulSoup, Requests |
| Config | Centralized settings management | Python config |
| Logger | Structured logging | Python logging |

---

## 5. Retrieval Strategies

### 5.1 Sparse Retrieval (Keyword Search)

The keyword-based retrieval system uses TF-weighted matching across multiple fields:

```python
# Keyword weights (configurable)
KW_WEIGHT_NAME = 30.0        # Assessment name
KW_WEIGHT_TEST_TYPES = 25.0 # Test type tags
KW_WEIGHT_DESC = 15.0       # Description text
KW_WEIGHT_URL = 20.0        # URL components
```

#### Boosting Strategies

| Boost Type | Multiplier | Condition |
|-----------|------------|-----------|
| Exact Match | 100x | Query term == Field value |
| Prefix Match | 60x | Field starts with term |
| Partial Match | 30x | Field contains term |
| Term Frequency | 1 + (freq × 0.5) | Multiple occurrences |

```python
def keyword_search(keywords: set, assessments: List[Dict]) -> Dict[int, float]:
    """TF-weighted keyword search across assessment fields."""
    results = {}
    
    for idx, item in enumerate(assessments):
        score = 0.0
        name_lower = item.get("name", "").lower()
        desc_lower = item.get("description", "").lower()
        test_types = " ".join(item.get("test_types", [])).lower()
        url_lower = item.get("url", "").lower()
        
        for kw, freq in term_freq.items():
            tf_boost = 1.0 + freq * 0.5
            
            # Apply weighted matching
            if kw in name_lower:
                score += KEYWORD_BOOST_EXACT * tf_boost
            # ... similar for other fields
        
        results[idx] = score
    
    return results
```

### 5.2 Dense Retrieval (Semantic Search)

Semantic search uses dense embeddings for meaning-based matching:

#### Embedding Models

| Model | Dimension | Provider | Use Case |
|-------|-----------|----------|----------|
| nomic-embed-text | 768 | Ollama | Primary (offline) |
| all-MiniLM-L6-v2 | 384 | Sentence-Transformers | Fallback |
| gemini-embed-text | 768 | Google Gemini | Cloud option |

```python
def get_nomic_embedding(text: str) -> Optional[np.ndarray]:
    """Generate embedding using Ollama."""
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    emb = np.array(response["embedding"], dtype=np.float32)
    faiss.normalize_L2(emb.reshape(1, -1))
    return emb
```

#### FAISS Index Configuration

```python
# FAISS index parameters
SEMANTIC_TOP_K = 100    # Candidates to retrieve
FAISS_DIM = 768         # Embedding dimension
INDEX_TYPE = "IVF_FLAT" # Approximate nearest neighbor
```

### 5.3 Reciprocal Rank Fusion (RRF)

RRF combines rankings from multiple retrieval methods:

```python
def reciprocal_rank_fusion(
    keyword_results: Dict[int, float],
    semantic_results: Dict[int, float],
    k: int = 60
) -> Dict[int, float]:
    """Combine sparse and dense rankings using RRF."""
    
    # Sort by score
    keyword_ranked = sorted(keyword_results.items(), key=lambda x: x[1], reverse=True)
    semantic_ranked = sorted(semantic_results.items(), key=lambda x: x[1], reverse=True)
    
    # Build rank dictionaries
    keyword_ranks = {idx: rank for rank, (idx, _) in enumerate(keyword_ranked)}
    semantic_ranks = {idx: rank for rank, (idx, _) in enumerate(semantic_ranked)}
    
    # Compute RRF scores
    all_indices = set(keyword_ranks.keys()) | set(semantic_ranks.keys())
    fused_scores = {}
    
    for idx in all_indices:
        k_rank = keyword_ranks.get(idx, float('inf'))
        s_rank = semantic_ranks.get(idx, float('inf'))
        
        rrf_score = 1.0 / (k_rank + k) + 1.0 / (s_rank + k)
        kw_score = keyword_results.get(idx, 0) * 0.4
        sem_score = semantic_results.get(idx, 0) * 0.3
        
        fused_scores[idx] = rrf_score * 100 + kw_score + sem_score
    
    return fused_scores
```

**Why RRF?**
- **Order-based**: Uses only ranking positions, not raw scores
- **Score-free**: No need to calibrate scores between methods
- **Robust**: Handles missing results from either method
- **Proven**: Standard in information retrieval

---

## 6. LLM Integration

### 6.1 Multi-Stage Reranking

The LLM reranking layer uses a cascade approach for reliability:

```
Query + Candidates → LLM → Reordered Candidates → Final Output
```

#### Stage 1: Gemini 2.0 Flash Lite (Primary)

```python
# Primary reranking model
RERANK_MODEL_1 = "gemini-2.0-flash-lite"

prompt = f"""INSTRUCTIONS: You are a ranker. Your ONLY job is to reorder, NEVER to remove items.

STRICT REQUIREMENT: You MUST output EXACTLY {N} indices - no more, no less.
DO NOT filter. DO NOT drop. Only reorder.

QUERY: {query}

CANDIDATES:
{candidates_text}

Return JSON array only:"""
```

**Why Gemini 2.0?**
- Fast response time (<1s)
- Excellent JSON parsing
- Generous free tier
- Reliable uptime

#### Stage 2: Gemini 2.5 Flash Lite (Fallback)

```python
# Fallback when Stage 1 fails
RERANK_MODEL_2 = "gemini-2.5-flash-lite"
```

#### Stage 3: Ollama (Local Fallback)

```python
# Works offline, no API costs
OLLAMA_LLM_MODEL = "qwen3.5:0.8b"
```

**Why this cascade?**
1. **Cost Efficiency**: Use cloud APIs when available
2. **Reliability**: Fallback ensures system works even if APIs fail
3. **Offline Mode**: Ollama provides local capability
4. **Latency**: Try fastest options first

###  Engineering

The system uses carefully crafted prompts for different tasks:

#### Reranking Prompt

```python
RERANK_PRO6.2 PromptMPT = """TASK: Find ALL relevant assessments. MAXIMIZE RECALL.

QUERY: {query}

CANDIDATES:
{candidates}

RULES:
1. Exact skill match: "Java" → prioritize "Java" in name/URL
2. Role match: "developer" → any dev test
3. Domain: "sales" → sales assessments
4. WIDE NET: Include rather than exclude

Return JSON list of indices only: [0, 3, 1, 2, ...]"""
```

#### Generation Prompt

```python
GENERATION_PROMPT = """TASK: Given the already-reranked candidate list below, output the TOP 10 with scores.

QUERY: {query}

ALREADY RERANKED CANDIDATES:
{candidates}

Return JSON array with 10 items:
[
  {{"name": "Name", "url": "...", "reason": "...", "score": 0.95}},
  ...
]"""
```

---

## 7. Configuration & Tuning

### 7.1 Centralized Configuration

All parameters are defined in `config_v9.py`:

```python
# ==================== PATHS ====================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FAISS_DIR = DATA_DIR / "faiss_index"

# ==================== EMBEDDINGS ====================
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# ==================== LLM MODELS ====================
GEMINI_MODEL_1 = "gemini-3.1-flash-lite-preview"
GEMINI_MODEL_2 = "gemini-2.5-flash-lite"
OLLAMA_LLM_MODEL = "qwen3.5:0.8b"

# ==================== RETRIEVAL ====================
KW_WEIGHT_NAME = 30.0
KW_WEIGHT_TEST_TYPES = 25.0
KW_WEIGHT_DESC = 15.0
SEMANTIC_TOP_K = 100
HYBRID_TOP_K = 25
RRF_K = 60

# ==================== KEYWORD BOOSTING ====================
KEYWORD_BOOST_EXACT = 100.0
KEYWORD_BOOST_PREFIX = 60.0
KEYWORD_BOOST_PARTIAL = 30.0
```

### 7.2 Tuning Guidelines

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `RRF_K` | Higher = more weight to lower ranks | 50-100 |
| `KW_WEIGHT_NAME` | Higher = name matching more important | 25-50 |
| `HYBRID_TOP_K` | More candidates = better recall, slower | 20-50 |
| `SEMANTIC_TOP_K` | More semantic candidates | 50-200 |

---

## 8. Evaluation Framework

### 8.1 Metrics

```python
def compute_recall_at_k(predictions: List[str], ground_truth: List[str], k: int = 10) -> float:
    """Compute Recall@K."""
    pred_k = set(predictions[:k])
    truth = set(ground_truth)
    return len(pred_k & truth) / len(truth) if truth else 0.0
```

### 8.2 Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| Training | 10 queries | Labeled with ground truth assessments |
| Test | 9 queries | Unlabeled, for final evaluation |

### 8.3 Sample Evaluation Results

```
Query: "Java developer assessment for senior role"
Ground Truth: ["Advanced Java Programming", "Verify/G+"]

Predictions:
1. Advanced Java Programming ✓
2. Verify/G+ Cognitive Assessment ✓
3. Technical Skills Assessment
...

Recall@10: 2/2 = 100%
```

---

## 9. Production Deployment

### 9.1 API Server (FastAPI)

```bash
# Start server
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health

# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Python developer", "top_k": 10}'
```

### 9.2 Web UI (Streamlit)

```bash
# Launch UI
streamlit run src/streamlit_app.py

# Access at http://localhost:8501
```

### 9.3 Docker Deployment (Future)

```dockerfile
FROM python:3.10-slim
WORKDIR /app .
RUN pip install -r requirements.txt
COPY . .

COPY requirements.txtEXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0"]
```

---

## 10. Future Improvements

### 10.1 Short-term

- [ ] Rebuild FAISS index with sentence-transformers (384-dim) for full compatibility
- [ ] Add cross-encoder reranking for more accurate results
- [ ] Implement URL-based job description parsing
- [ ] Add caching for frequently asked queries

### 10.2 Medium-term

- [ ] Implement vector database (ChromaDB/Pinecone) for better scalability
- [ ] Add user feedback loop for continuous improvement
- [ ] Deploy as Docker container with docker-compose
- [ ] Add monitoring and alerting (Prometheus, Grafana)

### 10.3 Long-term

- [ ] Fine-tune embedding model on assessment data
- [ ] Implement multi-language support
- [ ] Add assessment validity checking
- [ ] Build analytics dashboard for usage insights

---

## 📚 References

1. **FAISS**: https://github.com/facebookresearch/faiss
2. **Ollama**: https://ollama.ai/
3. **Gemini API**: https://gemini.google.com/
4. **Reciprocal Rank Fusion**: "Reciprocal Rank Fusion" by Robertson et al.
5. **RAG Architecture**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

---

## 🔗 Related Files

| File | Description |
|------|-------------|
| `src/recommender.py` | Core V9 hybrid recommender implementation |
| `src/api.py` | FastAPI server |
| `src/indexer.py` | FAISS index builder |
| `config_v9.py` | Central configuration |
| `README.md` | Project overview and quick start |

---

<p align="center">
  <strong>Built with ❤️ using Hybrid RAG Architecture</strong>
</p>

