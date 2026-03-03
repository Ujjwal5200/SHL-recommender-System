# SHL Assessment Recommendation System

An intelligent recommendation system that recommends relevant SHL assessments based on natural language queries or job descriptions.

## Problem Statement

Hiring managers and recruiters often struggle to find the right assessments for the roles they are hiring for. The current system relies on keyword searches and filters, making the process time-consuming and inefficient. This system uses AI/ML techniques to recommend relevant SHL assessments from a catalog of 377+ individual test solutions based on:
- Natural language queries
- Job descriptions (JD)
- Job Description URLs

## Project Overview

**Objective**: Build a recommendation system that suggests relevant SHL assessments (minimum 5, maximum 10) from a catalog of 377+ individual test solutions based on natural language queries or job descriptions.

**Key Requirements (from PDF)**:
- ✅ Crawl SHL product catalog and obtain at least 377 Individual Test Solutions
- ✅ Exclude "Pre-packaged Job Solutions" category
- ✅ Each recommendation includes: Assessment Name, URL
- ✅ Return minimum 5 to maximum 10 recommendations
- ✅ Provide API endpoint for programmatic access
- ✅ Calculate Mean Recall@K as evaluation metric

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SHL Assessment Recommendation System                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        USER INPUT                                │    │
│  │  • Natural Language Query (e.g., "I need Java developer")      │    │
│  │  • Job Description Text                                         │    │
│  │  • JD URL (optional)                                            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                   │                                      │
│                                   ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     API Layer (FastAPI)                          │    │
│  │  • GET  /health        - Health check endpoint                 │    │
│  │  • POST /recommend    - Get recommendations (JSON response)     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                   │                                      │
│                                   ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                Recommendation Engine (Hybrid)                     │    │
│  │                                                                  │    │
│  │  ┌──────────────────┐    ┌──────────────────┐                 │    │
│  │  │ Keyword Matching │    │ Semantic Search   │                 │    │
│  │  │                  │    │                  │                 │    │
│  │  │ • 70+ keywords  │    │ • FAISS Vector   │                 │    │
│  │  │ • Job titles    │    │ • Ollama Embed   │                 │    │
│  │  │ • Skills        │    │ • Top-50 search  │                 │    │
│  │  │ • Programming   │    │                  │                 │    │
│  │  │ • Test types    │    │                  │                 │    │
│  │  └────────┬─────────┘    └────────┬─────────┘                 │    │
│  │           │                        │                           │    │
│  │           └──────────┬─────────────┘                           │    │
│  │                      ▼                                         │    │
│  │  ┌──────────────────────────────────────────────────────┐      │    │
│  │  │              Combined Scoring & Ranking               │      │    │
│  │  │  • Keyword Score: 2.0x weight                        │      │    │
│  │  │  • Semantic Score: 1.0x weight                       │      │    │
│  │  └──────────────────────────────────────────────────────┘      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                   │                                      │
│                                   ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                       Data Layer                                 │    │
│  │                                                                  │    │
│  │  ┌─────────────────────┐  ┌─────────────────────────────────┐  │    │
│  │  │  SHL Assessments    │  │     FAISS Vector Index          │  │    │
│  │  │  (377+ tests)      │  │     (Ollama Embeddings)        │  │    │
│  │  │  • Name            │  │     • nomic-embed-text          │  │    │
│  │  │  • URL            │  │     • 768 dimensions            │  │    │
│  │  │  • Test Types     │  │                                 │  │    │
│  │  │  • Description    │  │                                 │  │    │
│  │  │  • Duration      │  │                                 │  │    │
│  │  │  • Remote/Adaptive│  │                                 │  │    │
│  │  └─────────────────────┘  └─────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Metrics - Mean Recall@K

As specified in the PDF, Mean Recall@K is the evaluation metric. Here's how it's calculated:

### Formula
```
Recall@K = (Number of relevant assessments in top K) / (Total relevant assessments for the query)

Mean Recall@K = (1/N) × Σ Recall@K  (for i = 1 to N)
```

Where:
- K = Number of recommendations to consider (5 or 10)
- N = Total number of test queries (10)
- Relevant assessments = Ground truth labels from Gen_AI_Dataset.xlsx

### Ground Truth Data (from Gen_AI_Dataset.xlsx)

| Query | Total Relevant |
|-------|---------------|
| Java Developer | 5 |
| Sales Graduate | 9 |
| COO (China) | 6 |
| Sound Manager | 5 |
| Content Writer | 5 |
| QA Engineer | 9 |
| Assistant Admin | 6 |
| Marketing Manager | 5 |
| Consultant JD | 5 |
| Data Analyst | 10 |

### Final Results

| Metric | Score |
|--------|-------|
| **Mean Recall@5** | **0.2133 (21.33%)** |
| **Mean Recall@10** | **0.2844 (28.44%)** |

### Per-Query Results

| Query | Relevant | In Top 5 | In Top 10 | Recall@5 | Recall@10 |
|-------|----------|----------|-----------|----------|-----------|
| Java Developer | 5 | 3 | 3 | 60.00% | 60.00% |
| Sales Graduate | 9 | 1 | 1 | 11.11% | 11.11% |
| COO (China) | 6 | 0 | 0 | 0.00% | 0.00% |
| Sound Manager | 5 | 2 | 2 | 40.00% | 40.00% |
| Content Writer | 5 | 1 | 3 | 20.00% | 60.00% |
| QA Engineer | 9 | 2 | 3 | 22.22% | 33.33% |
| Assistant Admin | 6 | 0 | 0 | 0.00% | 0.00% |
| Marketing Manager | 5 | 1 | 1 | 20.00% | 20.00% |
| Consultant JD | 5 | 0 | 0 | 0.00% | 0.00% |
| Data Analyst | 10 | 4 | 6 | 40.00% | 60.00% |

---

## LLM Experiments

We experimented with LLM-based reranking using Gemma 3 1b (Ollama):

| Approach | Mean Recall@5 | Mean Recall@10 |
|----------|---------------|---------------|
| **Hybrid (No LLM)** | **21.33%** | **28.44%** |
| Hybrid + LLM Boost | 20.33% | 26.44% |
| Hybrid + LLM Rerank | 21.33% | 25.33% |

**Conclusion**: The baseline hybrid approach (keyword + semantic) outperforms LLM-based approaches with Gemma 3 1b. The keyword matching and semantic search combination provides better results for this task.

---

## Sample Queries (from PDF Appendix 1)

1. **Java Developer**: "I am hiring for Java developers who can also collaborate effectively with my business teams."
2. **Multi-Skill**: "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script."
3. **Analyst**: "Here is a JD text, can you recommend some assessment that can help me screen applications. I am hiring for an analyst and wants applications to screen using Cognitive and personality tests"

---

## Project Structure

```
SHL-assessment-recommender/
├── src/
│   ├── api.py                    # FastAPI endpoints
│   ├── app.py                    # Streamlit web UI
│   ├── config.py                 # Configuration settings
│   ├── recommender_final.py      # Hybrid recommendation engine
│   ├── indexer.py               # FAISS index builder
│   ├── scraper.py               # SHL catalog scraper
│   └── logger.py                # Logging utility
├── data/
│   ├── shl_individual_tests_20260302_1257.json  # 377 SHL assessments
│   ├── faiss_index/             # Vector store index
│   │   ├── index.faiss
│   │   └── index.pkl
│   ├── train_predictions.csv    # Training data (10 queries)
│   ├── test_predictions.csv     # Test predictions (10 queries)
│   └── Gen_AI_Dataset.xlsx     # Ground truth data
├── tests/
│   └── test-recommender.py     # Unit tests
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── report_approach.md          # Detailed approach report
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- Ollama (for local embeddings)

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Setup Ollama & Pull Embedding Model
```
ollama pull nomic-embed-text
```

### 3. Start the API Server
```
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 4. Start the Web UI (optional)
```
streamlit run src/app.py
```

---

## API Usage

### Health Check
```
GET /health
```
Response: `{"status": "healthy"}`

### Recommendation Endpoint
```
POST /recommend

Request:
{
  "query": "I am hiring for Java developers who can also collaborate effectively with my business teams."
}

Response:
{
  "recommended_assessments": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/...",
      "test_types": ["K"],
      "duration_minutes": 30,
      "adaptive_support": "No",
      "remote_support": "Yes",
      "description": "..."
    }
  ]
}
```

---

## Submission Format

The test predictions CSV follows the exact format specified in Appendix 3 of the PDF:

```
csv
query,Assessment_url
Query 1,https://www.shl.com/products/product-catalog/view/assessment-1/
Query 1,https://www.shl.com/products/product-catalog/view/assessment-2/
Query 1,https://www.shl.com/products/product-catalog/view/assessment-3/
Query 2,https://www.shl.com/products/product-catalog/view/assessment-4/
...
```

---

## Key Features Implemented

### ✅ Data Pipeline
- Web scraper to crawl SHL product catalog
- Extracts 377+ Individual Test Solutions
- Parses and structures product data

### ✅ Technology Stack
- FastAPI for REST API
- Streamlit for Web UI
- FAISS for vector storage
- Ollama for embeddings

### ✅ Hybrid Recommendation
- Keyword matching with priority scoring (URL: 5.0, Name: 4.0, Test Types: 2.5, Description: 1.0)
- Semantic search with vector embeddings (50 candidates)
- Combined scoring (Keyword: 2x, Semantic: 1x)

### ✅ Evaluation
- Mean Recall@K calculated using Gen_AI_Dataset.xlsx ground truth
- Results: Recall@5 = 21.33%, Recall@10 = 28.44%

---

## License

This project is for educational purposes as part of the SHL AI Intern Assessment.
