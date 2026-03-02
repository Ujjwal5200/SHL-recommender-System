# SHL Assessment Recommendation System

An intelligent recommendation system that recommends relevant SHL assessments based on natural language queries or job descriptions.

## Problem Statement

Hiring managers and recruiters often struggle to find the right assessments for the roles they are hiring for. This system uses AI/ML techniques to recommend relevant SHL assessments from a catalog of 377+ individual test solutions based on natural language queries or job descriptions.

## Features

- **Natural Language Query Processing**: Accept job descriptions or requirements in natural language
- **Hybrid Recommendation Engine**: Combines keyword matching + semantic search using vector embeddings
- **REST API**: FastAPI endpoint for programmatic access
- **Web UI**: Streamlit-based user interface
- **Balanced Recommendations**: Returns relevant assessments balanced across different test types (Knowledge, Skills, Personality, etc.)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SHL Recommendation System                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Web UI    │    │    API     │    │   Tests     │     │
│  │ (Streamlit)│    │ (FastAPI)  │    │ (Pytest)    │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│              ┌─────────────────────────┐                   │
│              │   Recommender Engine    │                   │
│              │  (Hybrid: Keyword +     │                   │
│              │   Semantic Search)      │                   │
│              └───────────┬─────────────┘                   │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              FAISS Vector Index                     │   │
│  │         (Ollama Embeddings - nomic-embed-text)      │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         377 SHL Individual Test Solutions          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
SHL-assessment-recommender/
├── src/
│   ├── api.py              # FastAPI endpoints (/health, /recommend)
│   ├── app.py              # Streamlit web UI
│   ├── config.py           # Configuration settings
│   ├── recommender_final.py # Hybrid recommender engine
│   ├── indexer.py          # FAISS index builder
│   ├── scraper.py          # SHL catalog scraper
│   └── logger.py           # Logging utility
├── data/
│   ├── shl_individual_tests_20260302_1257.json  # 377 SHL assessments
│   ├── faiss_index/        # Vector store index
│   └── Gen_AI_Dataset.xlsx # Ground truth data (10 queries)
├── test_predictions.csv    # Predictions for test set (100 rows)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Technology Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Vector Store**: FAISS
- **Embeddings**: Ollama (nomic-embed-text model)
- **LLM Integration**: Gemini (optional, for advanced reranking)
- **Data Processing**: LangChain, LangChain Community
- **Evaluation**: Pandas, Scikit-learn

## Setup

1. Install dependencies:
```
bash
pip install -r requirements.txt
```

2. Ensure Ollama is running with the embedding model:
```
bash
ollama pull nomic-embed-text
```

3. Start the API:
```
bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

4. Start the web UI:
```
bash
streamlit run src/app.py
```

## API Endpoints

### Health Check
```
GET /health
```
Returns: `{"status": "healthy"}`

### Recommend Assessments
```
POST /recommend
```
Request:
```
json
{
  "query": "I am hiring for Java developers with strong collaboration skills"
}
```

Response:
```
json
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

## Evaluation Results

- **Mean Recall@5**: 0.213
- **Mean Recall@10**: 0.284

### Performance Metrics by Query Type

| Query Type | Recall@5 | Recall@10 |
|------------|----------|-----------|
| Java Developer | 0.600 | 0.600 |
| Sales Graduate | 0.111 | 0.111 |
| COO (China) | 0.000 | 0.000 |
| Content Writer | 0.200 | 0.400 |
| Data Analyst | 0.222 | 0.333 |

## Submission Format

The test predictions CSV file follows this format:

```
csv
query,Assessment_url
Query 1,https://www.shl.com/products/product-catalog/view/...
Query 1,https://www.shl.com/products/product-catalog/view/...
Query 2,https://www.shl.com/products/product-catalog/view/...
...
```

## Key Features Implemented

1. **Data Pipeline**: Scraped 377 SHL individual test solutions from catalog
2. **Hybrid Retrieval**: Combined keyword matching + semantic search
3. **Vector Index**: FAISS index built with Ollama embeddings
4. **Evaluation Framework**: Mean Recall@K metrics implemented
5. **Multi-domain Support**: Balanced recommendations across test types

## Future Improvements

- Implement LLM-based reranking for better precision
- Add support for JD URL parsing
- Improve recall for low-performing query categories
- Deploy to cloud for public access
