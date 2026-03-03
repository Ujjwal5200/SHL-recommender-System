# SHL Assessment Recommendation System - Detailed Approach Report

## Executive Summary

This report documents the approach taken to build an intelligent recommendation system for SHL assessments. The system uses a **hybrid approach** combining keyword matching and semantic search with vector embeddings to recommend relevant assessments from a catalog of 377+ Individual Test Solutions.

**Key Results:**
- **Mean Recall@5: 0.2133 (21.33%)**
- **Mean Recall@10: 0.2844 (28.44%)**
- Successfully crawled 377+ assessments from SHL catalog
- Implemented FastAPI with `/recommend` endpoint

---

## 1. Problem Understanding

### 1.1 Task Overview

The objective is to build a recommendation system that:
1. Takes natural language queries or job descriptions as input
2. Recommends 5-10 most relevant SHL Individual Test Solutions
3. Excludes "Pre-packaged Job Solutions" category
4. Each recommendation includes: Assessment Name, URL, Test Types

### 1.2 Evaluation Criteria (from PDF)

The solution was evaluated on:
- **Solution Approach**: Well-defined pipeline connecting data ingestion, retrieval, and recommendation
- **Data Pipeline**: Web scraping, clean parsing, efficient storage using embeddings/vector databases
- **Technology Stack**: Modern LLM-based or retrieval-augmented techniques
- **Evaluation**: Mean Recall@K implementation
- **Performance**: Recommendation accuracy measured by Mean Recall@10

---

## 2. Solution Architecture

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER LAYER                                      │
│   Natural Language Query → Job Description → JD URL                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         API LAYER (FastAPI)                             │
│   GET /health          → Health Check                                   │
│   POST /recommend     → Recommendation Engine                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    RECOMMENDATION ENGINE                                 │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    HYBRID RETRIEVAL                              │   │
│   │                                                                  │   │
│   │   ┌─────────────────────┐      ┌─────────────────────┐       │   │
│   │   │  KEYWORD MATCHING   │      │  SEMANTIC SEARCH    │       │   │
│   │   │                     │      │                     │       │   │
│   │   │ • 70+ Keywords     │      │ • FAISS Index       │       │   │
│   │   │ • Job Titles      │      │ • Ollama Embeddings │       │   │
│   │   │ • Skills          │      │ • Top-50 Search     │       │   │
│   │   │ • Test Types     │      │                     │       │   │
│   │   │                     │      │                     │       │   │
│   │   │ Scoring:           │      │ Scoring:            │       │   │
│   │   │ • URL: 5.0        │      │ • 1.0 - (i*0.018) │       │   │
│   │   │ • Name: 4.0       │      │                     │       │   │
│   │   │ • Test Types: 2.5 │      │                     │       │   │
│   │   │ • Description: 1.0│      │                     │       │   │
│   │   └──────────┬──────────┘      └──────────┬──────────┘       │   │
│   │              │                            │                   │   │
│   │              └────────────┬───────────────┘                   │   │
│   │                           ▼                                   │   │
│   │   ┌─────────────────────────────────────────────────────┐    │   │
│   │   │              SCORE COMBINATION                       │    │   │
│   │   │  Final Score = (Keyword × 2.0) + (Semantic × 1.0)│    │   │
│   │   └─────────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                      │
│                                                                          │
│   ┌─────────────────────────────┐    ┌─────────────────────────────┐   │
│   │    SHL CATALOG (JSON)      │    │   FAISS VECTOR INDEX       │   │
│   │                             │    │                             │   │
│   │  377+ Assessments          │    │  377 vectors × 768 dims    │   │
│   │  • name: string            │    │  Built with:              │   │
│   │  • url: string            │    │  • Ollama                 │   │
│   │  • test_types: string[]  │    │  • nomic-embed-text       │   │
│   │  • description: string    │    │                             │   │
│   │  • duration_minutes: int  │    │                             │   │
│   │  • adaptive_support      │    │                             │   │
│   │  • remote_support        │    │                             │   │
│   └─────────────────────────────┘    └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Details

### 3.1 Data Pipeline

#### Web Scraping (`src/scraper.py`)
- Used requests + BeautifulSoup for HTML parsing
- Pagination handling (12 items per page, up to 32 pages)
- Detail page enrichment (description, duration extraction)
- Rate limiting (1.4s between requests)
- ✅ Successfully crawled 377+ Individual Test Solutions

#### Index Building (`src/indexer.py`)
- LangChain Document creation
- Ollama embeddings (nomic-embed-text)
- FAISS vector store

### 3.2 Recommendation Engine (`src/recommender_final.py`)

#### Keyword Extraction
- 70+ Keywords including:
  - Programming languages: java, python, sql, javascript, c++, etc.
  - Job titles: developer, engineer, analyst, manager, etc.
  - Skills: communication, leadership, teamwork, etc.

#### Keyword Scoring
| Match Location | Weight |
|---------------|--------|
| URL Slug | 5.0 |
| Assessment Name | 4.0 |
| Test Types | 2.5 |
| Description | 1.0 |

#### Semantic Search
- FAISS similarity search with 50 candidates
- Position-based scoring: `1.0 - (i * 0.018)`

#### Score Combination
```
Final Score = (Keyword Score × 2.0) + (Semantic Score × 1.0)
```

---

## 4. Evaluation & Mean Recall@K Calculation

### 4.1 Mean Recall@K Formula (as specified in PDF)

```
Recall@K = (Number of relevant assessments in top K) / (Total relevant assessments for the query)

Mean Recall@K = (1/N) × Σ Recall@K (for i = 1 to N)
```

### 4.2 Ground Truth Analysis (from Gen_AI_Dataset.xlsx)

The dataset contains 65 labeled query-assessment pairs across 10 unique queries.

### 4.3 Final Results

| Metric | Score |
|--------|-------|
| **Mean Recall@5** | **0.2133 (21.33%)** |
| **Mean Recall@10** | **0.2844 (28.44%)** |

---

## 5. LLM Experiments

We experimented with LLM-based reranking using Gemma 3 1b from Ollama:

### 5.1 Approaches Tested

| Approach | Description | Recall@5 | Recall@10 |
|----------|-------------|----------|-----------|
| **Baseline Hybrid** | Keyword + Semantic only | **21.33%** | **28.44%** |
| LLM Boost | Add LLM score boost (+3 pts) | 20.33% | 26.44% |
| LLM Rerank | Full LLM-based reranking | 21.33% | 25.33% |

### 5.2 Findings

- The baseline hybrid approach (keyword + semantic) outperforms LLM-based approaches
- Gemma 3 1b may not be optimal for this task - smaller models may lack domain understanding
- Keyword matching provides strong precision for technical queries (Java, Python)
- Semantic search helps with contextual understanding but adds noise

---

## 6. Submission Materials

1. **API Endpoint**: FastAPI at `/recommend`
2. **Code Repository**: Complete source code
3. **Web Application**: Streamlit UI
4. **Predictions CSV**: `test_predictions.csv` (100 rows, 10 queries × 10 recommendations)

---

## 7. Conclusion

The hybrid recommendation system successfully combines:
- **Keyword matching** for precise skill/exact matches
- **Semantic search** for contextual understanding
- **Combined scoring** for balanced results

The evaluation shows **Mean Recall@10 of 28.44%**, demonstrating the system's ability to recommend relevant assessments from the SHL catalog based on natural language queries.

---

## Appendix: Files

- `README.md` - Project overview and quick start
- `report_approach.md` - This detailed approach report
- `data/Gen_AI_Dataset.xlsx` - Ground truth data for evaluation
- `data/test_predictions.csv` - Model predictions on test set
