# SHL Assessment Recommendation System - Approach Report

## Executive Summary

Built an intelligent recommendation system for SHL assessments using a hybrid approach combining keyword matching and semantic search with vector embeddings. The system achieves Mean Recall@5 of 0.213 and Mean Recall@10 of 0.284 on the provided test set.

---

## 1. Problem Understanding

**Objective**: Build a recommendation system that suggests relevant SHL assessments (5-10) from a catalog of 377+ individual test solutions based on:
- Natural language queries
- Job descriptions
- Job Description URLs

**Key Requirements**:
- Minimum 377 Individual Test Solutions (exclude Pre-packaged Job Solutions)
- Recommendations must include: Name, URL, Test Types
- Balanced recommendations across different assessment categories
- API endpoint for programmatic access

---

## 2. Solution Approach

### 2.1 Data Pipeline

1. **Web Scraping**: Built scraper to crawl SHL product catalog
2. **Data Extraction**: Extracted 377 individual test solutions with:
   - Assessment name
   - URL
   - Test types (K=Knowledge, P=Personality, A=Aptitude, etc.)
   - Description
   - Duration
   - Remote/Adaptive support

3. **Data Storage**: JSON file with structured assessment data

### 2.2 Index Building

1. **Text Processing**: Created rich document content combining:
   - Assessment name
   - Description (truncated to 1000 chars)
   - Test types
   - URL slug for better matching

2. **Embedding Generation**: Used Ollama with nomic-embed-text model

3. **Vector Index**: Built FAISS index for efficient similarity search

### 2.3 Recommendation Engine

**Hybrid Approach**:

```
Query → Keyword Matching + Semantic Search → Combined Scoring → Ranking
```

**Keyword Matching**:
- Comprehensive keyword dictionary (programming languages, job titles, skills)
- Priority scoring: URL slug > Name > Test Types > Description
- Multi-word phrase detection

**Semantic Search**:
- FAISS vector similarity search
- Top-30 candidates retrieved

**Combined Scoring**:
- Keyword score: 2.5x weight
- Semantic score: 1x weight
- Final ranking by total score

---

## 3. Implementation Details

### 3.1 Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI |
| Frontend | Streamlit |
| Vector Store | FAISS |
| Embeddings | Ollama (nomic-embed-text) |
| Data Processing | LangChain, LangChain Community |
| Evaluation | Pandas, Scikit-learn |

### 3.2 Key Files

- `src/recommender_final.py`: Hybrid recommendation engine
- `src/api.py`: FastAPI endpoints (/health, /recommend)
- `src/app.py`: Streamlit web UI
- `src/indexer.py`: FAISS index builder
- `src/scraper.py`: SHL catalog scraper

---

## 4. Evaluation

### 4.1 Metrics

Used **Mean Recall@K** as primary metric:

```
Recall@K = (Relevant items in top K) / (Total relevant items)
Mean Recall@K = Average Recall@K across all queries
```

### 4.2 Results

| Metric | Score |
|--------|-------|
| Mean Recall@5 | 0.213 |
| Mean Recall@10 | 0.284 |

### 4.3 Per-Query Analysis

| Query | Relevant | Recall@5 | Recall@10 |
|-------|----------|----------|-----------|
| Java Developer | 5 | 0.600 | 0.600 |
| Sales Graduate | 9 | 0.111 | 0.111 |
| COO (China) | 6 | 0.000 | 0.000 |
| Content Writer | 5 | 0.200 | 0.400 |
| 1-Hour Assessment | 9 | 0.222 | 0.333 |
| Assistant Admin | 6 | 0.000 | 0.000 |
| Marketing Manager | 5 | 0.000 | 0.200 |
| Consultant JD | 5 | 0.400 | 0.400 |
| Data Analyst | 5 | 0.200 | 0.400 |
| Sound Manager | 5 | 0.400 | 0.400 |

---

## 5. Optimization Efforts

### 5.1 Initial Baseline
- Basic keyword matching: Recall@5: 0.15, Recall@10: 0.20

### 5.2 Improvements Made

1. **Enhanced Keyword Dictionary**: Added comprehensive keywords
2. **URL Slug Matching**: Prioritized matches in URL slugs
3. **Multi-word Phrases**: Added phrase detection
4. **Hybrid Scoring**: Combined keyword + semantic search
5. **Semantic Expansion**: Increased candidates from 20 to 30

### 5.3 Final Results

- Mean Recall@5: **0.213** (+42% improvement)
- Mean Recall@10: **0.284** (+42% improvement)

---

## 6. Challenges & Limitations

1. **Ground Truth Quality**: Some queries have ambiguous relevance labels
2. **Query Complexity**: Long JDs with multiple requirements harder to match
3. **LLM Quotas**: Gemini API quota exhausted, limiting advanced reranking
4. **Semantic Gap**: Vector embeddings may miss exact keyword matches

---

## 7. Submission Materials

1. **API Endpoint**: FastAPI at `/recommend` (JSON response)
2. **Code Repository**: GitHub (complete source code)
3. **Web Application**: Streamlit UI at `/app`
4. **Predictions CSV**: `test_predictions.csv` (100 rows, 10 queries × 10 recommendations)

---

## 8. Conclusion

The hybrid recommendation system successfully combines keyword matching with semantic search to provide relevant SHL assessment recommendations. While performance meets basic requirements, there is room for improvement through:
- LLM-based reranking
- Better query understanding
- Enhanced evaluation metrics (NDCG, MAP)
