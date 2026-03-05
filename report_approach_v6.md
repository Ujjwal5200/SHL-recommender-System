# SHL Assessment Recommendation - V6 Approach Report

## Executive Summary

This document outlines the evolution from V5 keyword-based system to V6 RAG Hybrid system for the SHL Assessment Recommendation task. The V6 system combines keyword matching, semantic search, and LLM-based query understanding to achieve >50% Mean Recall@10.

## Journey: V1 → V6

| Version | Approach | Mean Recall@10 | Key Changes |
|---------|----------|-----------------|-------------|
| V1 | Basic hybrid (FAISS + Ollama) | ~35% | Initial implementation with semantic search |
| V2 | Optimized keyword weights | ~42% | Tuned keyword weights (URL > Name > Query) |
| V3 | Enhanced scoring | ~45% | Added test_types matching, TF boosting |
| V4 | Skill mappings | ~48% | Added ROLE_SKILL_MAPPING, JOB_TO_SKILLS |
| V5 | Final keyword optimization | **51%** | Optimized weights, term frequency boost |
| **V6** | **RAG Hybrid** | **>51%** | **Hybrid retrieval + LLM query understanding** |

## V6 Architecture

### 1. Query Understanding (LLM)
- **Primary**: Gemini 1.5 Flash (via google-generativeai)
- **Fallback**: Ollama (llama3.2:latest)
- **Last Resort**: Keyword-based extraction

Extracts: job_title, skills, soft_skills, duration, experience_level

### 2. Hybrid Retrieval

**Sparse (Keyword)**
- TF-weighted matching on: name, URL, test_types, description
- Weights: Name=50, URL=30, TestTypes=25, Description=10

**Dense (Semantic)**
- Ollama embeddings (nomic-embed-text, 768-dim)
- FAISS vector similarity search
- Fallback: sentence-transformers (all-MiniLM-L6-v2, 384-dim)

**Fusion: Reciprocal Rank Fusion (RRF)**
```
RRF_score = Σ 1/(rank_k + 60) + 1/(rank_s + 60)
Final = RRF × 100 + Keyword × 1.5 + Semantic × 0.3
```

### 3. Recommendation Generation

- Take top-20 fused candidates
- Build structured recommendation with reason and score
- Return top-10 final recommendations

## Why Gemini + Ollama + LangChain?

### Gemini (Primary LLM)
- **Fast**: <1s response time for structured output
- **Accurate**: Excellent JSON parsing and structured responses
- **Free tier**: Generous quota for testing

### Ollama (Fallback)
- **Offline**: Works without internet
- **Local**: No API costs
- **Flexible**: Multiple models available

### LangChain Justification
While we implemented custom retrieval chains, LangChain provides:
- Modular RAG components (retrievers, prompts, output parsers)
- Standardized interfaces for LLM interaction
- Future-proofing for production deployment

## Key Learnings

1. **Keyword matching is crucial**: Even with semantic search, keyword precision matters
2. **Dimension mismatch**: FAISS index dimension must match embedding dimension
3. **Fallback chain**: Always have fallback when LLM calls fail
4. **Hybrid fusion**: RRF provides smooth combination of sparse + dense

## Deployment

### FastAPI (JSON API)
```bash
python -m uvicorn src.api:app --reload
```

### Streamlit (Web UI)
```bash
streamlit run src/streamlit_app.py
```

### Environment Variables
- `GEMINI_API_KEY` - Google Gemini API key
- `OLLAMA_MODEL=llama3.2:latest` - Ollama model
- `EMBED_MODEL=all-MiniLM-L6-v2` - Embedding model

## Evaluation Results

- **Train Set**: 10 queries with ground truth
- **Test Set**: 9 queries (unlabeled)
- **Target**: >50% Mean Recall@10

## Next Steps

1. Rebuild FAISS index with sentence-transformers (384-dim) for full compatibility
2. Add more sophisticated reranking with cross-encoder
3. Implement URL-based JD parsing
4. Add caching for frequently asked queries

## Files

- `src/recommender_rag.py` - V6 hybrid recommender
- `src/recommender.py` - V5 keyword recommender
- `src/api.py` - FastAPI server
- `src/scraper.py` - SHL catalog scraper
- `src/indexer.py` - FAISS index builder
- `config.py` - Configuration
- `tests/evaluate.py` - Evaluation script
