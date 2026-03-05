"""
SHL Assessment Recommender - V9 PRODUCTION READY
Uses config_v9.py for all settings
"""
import os
import json
import re
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

# Import configuration
from src.config_v9 import (
    FAISS_DIR, DATA_DIR, LOG_LEVEL,
    OLLAMA_EMBED_MODEL, OLLAMA_LLM_MODEL, 
    GEMINI_MODEL_1, GEMINI_MODEL_2, RERANK_MODEL_1, RERANK_MODEL_2,
    GEMINI_MODEL, RERANK_MODEL, RERANK_CANDIDATES,
    SEMANTIC_TOP_K, HYBRID_TOP_K, RRF_K,
    KEYWORD_BOOST_EXACT, KEYWORD_BOOST_PREFIX, KEYWORD_BOOST_PARTIAL,
    KW_WEIGHT_NAME, KW_WEIGHT_TEST_TYPES, KW_WEIGHT_DESC,
    STOPWORDS, logger
)

# Load environment
from dotenv import load_dotenv
load_dotenv()

# ==================== LLM CLIENTS ====================
GEMINI_AVAILABLE = False
GEMINI_CLIENT = None

try:
    import google.genai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        GEMINI_CLIENT = genai.Client(api_key=api_key)
        GEMINI_AVAILABLE = True
        logger.info("GEMINI_AVAILABLE: True")
except Exception as e:
    logger.warning(f"Google GenAI init failed: {e}")

OLLAMA_AVAILABLE = False
try:
    import ollama
    OLLAMA_AVAILABLE = True
    logger.info("OLLAMA_AVAILABLE: True")
except:
    logger.warning("Ollama not available")


# ==================== DATA LOADING ====================

def load_assessments() -> List[Dict]:
    """Load assessments from FAISS metadata."""
    meta_path = FAISS_DIR / "metadata.json"
    if not meta_path.exists():
        from glob import glob
        files = sorted(glob(str(DATA_DIR / "shl_individual_tests_*.json")), reverse=True)
        if files:
            with open(files[0], encoding="utf-8") as f:
                assessments = json.load(f)
                logger.info(f"Loaded {len(assessments)} from catalog")
                return assessments
        raise FileNotFoundError("No assessment data found")
    
    with open(meta_path, encoding="utf-8") as f:
        assessments = json.load(f)
    logger.info(f"Loaded {len(assessments)} assessments")
    return assessments


def load_faiss_index() -> Tuple[faiss.Index, np.ndarray]:
    """Load FAISS index."""
    index_path = FAISS_DIR / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found")
    
    index = faiss.read_index(str(index_path))
    logger.info(f"Loaded FAISS: {index.ntotal} vectors, dim {index.d}")
    return index, np.empty((0, index.d))


# ==================== KEYWORD EXTRACTION ====================

def extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text."""
    words = text.lower().split()
    keywords = set()
    for w in words:
        w = re.sub(r'[^a-z0-9+-]', '', w)
        if len(w) > 2 and w not in STOPWORDS:
            keywords.add(w)
    return keywords


def get_term_frequency(text: str, keywords: set) -> Dict[str, int]:
    """Get term frequency."""
    words = text.lower().split()
    freq = Counter()
    for w in words:
        w = re.sub(r'[^a-z0-9+-]', '', w)
        if w in keywords:
            freq[w] += 1
    return dict(freq)


# ==================== KEYWORD SEARCH ====================

def keyword_search(keywords: set, assessments: List[Dict], term_freq: Dict[str, int]) -> Dict[int, float]:
    """Keyword-based search."""
    results = {}
    
    for idx, item in enumerate(assessments):
        score = 0.0
        name_lower = item.get("name", "").lower()
        desc_lower = item.get("description", "").lower()
        test_types = " ".join(item.get("test_types", [])).lower()
        url_lower = item.get("url", "").lower()
        
        for kw, freq in term_freq.items():
            tf_boost = 1.0 + freq * 0.5
            
            # Name matching
            if kw in name_lower:
                if name_lower == kw:
                    score += KEYWORD_BOOST_EXACT * tf_boost
                elif name_lower.startswith(kw + '-') or name_lower.startswith(kw + ' '):
                    score += KEYWORD_BOOST_PREFIX * tf_boost
                elif '-' + kw in name_lower:
                    score += 50 * tf_boost
                else:
                    score += KEYWORD_BOOST_PARTIAL * tf_boost
            
            # Test types
            if kw in test_types:
                score += KW_WEIGHT_TEST_TYPES * tf_boost
            
            # URL
            if kw in url_lower:
                score += 20 * tf_boost
            
            # Description
            if kw in desc_lower:
                score += KW_WEIGHT_DESC * tf_boost
        
        if score > 0:
            results[idx] = score
    
    return results


# ==================== SEMANTIC SEARCH ====================

def get_nomic_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding using Ollama."""
    if not OLLAMA_AVAILABLE:
        return None
    
    try:
        response = ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
        emb = np.array(response["embedding"], dtype=np.float32)
        faiss.normalize_L2(emb.reshape(1, -1))
        return emb
    except Exception as e:
        logger.warning(f"NomicEmbed error: {e}")
        return None


def semantic_search(query: str, index: faiss.Index, assessments: List[Dict], top_k: int = None) -> Dict[int, float]:
    """Semantic search."""
    if top_k is None:
        top_k = SEMANTIC_TOP_K
    
    query_emb = get_nomic_embedding(query)
    if query_emb is None:
        return {}
    
    search_k = min(top_k * 2, len(assessments))
    distances, indices = index.search(query_emb.reshape(1, -1), search_k)
    
    results = {}
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(assessments):
            results[idx] = (search_k - rank) * 0.5 + dist * 10
    
    return results


# ==================== HYBRID RETRIEVAL ====================

def reciprocal_rank_fusion(keyword_results: Dict[int, float], semantic_results: Dict[int, float], k: int = None) -> Dict[int, float]:
    """Combine keyword and semantic results."""
    if k is None:
        k = RRF_K
    
    keyword_ranked = sorted(keyword_results.items(), key=lambda x: x[1], reverse=True)
    semantic_ranked = sorted(semantic_results.items(), key=lambda x: x[1], reverse=True)
    
    keyword_ranks = {idx: rank for rank, (idx, _) in enumerate(keyword_ranked)}
    semantic_ranks = {idx: rank for rank, (idx, _) in enumerate(semantic_ranked)}
    
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


def hybrid_retrieve(query: str, assessments: List[Dict], index: faiss.Index, top_k: int = None) -> List[Tuple[int, float]]:
    """Hybrid retrieval."""
    if top_k is None:
        top_k = HYBRID_TOP_K
    
    keywords = extract_keywords(query)
    term_freq = get_term_frequency(query, keywords)
    
    keyword_results = keyword_search(keywords, assessments, term_freq)
    semantic_results = semantic_search(query, index, assessments)
    
    fused_scores = reciprocal_rank_fusion(keyword_results, semantic_results)
    
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


# ==================== OLLAMA LLM FUNCTIONS ====================

def generate_with_ollama(prompt: str) -> Optional[str]:
    """Generate response using Ollama LLM."""
    if not OLLAMA_AVAILABLE:
        return None
    
    try:
        response = ollama.chat(
            model=OLLAMA_LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.3}
        )
        return response['message']['content']
    except Exception as e:
        logger.warning(f"Ollama generation error: {e}")
        return None


def rerank_with_ollama(query: str, candidates: List[Dict]) -> List[Dict]:
    """Ollama reranking - fallback when Gemini fails."""
    if not candidates or not OLLAMA_AVAILABLE:
        return candidates[:10]
    
    # Build candidate info
    candidate_list = []
    for i, c in enumerate(candidates[:RERANK_CANDIDATES]):
        name = c.get('name', '')
        url = c.get('url', '')
        test_types = ', '.join(c.get('test_types', []))
        desc = c.get('description', '')[:150]
        
        info = f"""{i}. NAME: {name}
   URL: {url.split('/')[-1]}
   TYPES: {test_types}
   DESC: {desc}"""
        candidate_list.append(info)
    
    candidates_text = "\n\n".join(candidate_list)
    
    # Prompt for reranking
    prompt = f"""TASK: Find ALL relevant assessments. MAXIMIZE RECALL.

QUERY: {query}

CANDIDATES:
{candidates_text}

RULES:
1. Exact skill match: "Java" → prioritize "Java" in name/URL
2. Role match: "developer" → any dev test, "analyst" → analyst tests
3. Domain: "sales" → sales, "marketing" → marketing
4. WIDE NET: Include rather than exclude

Return JSON list of indices only: [0, 3, 1, 2, ...]"""

    try:
        response_text = generate_with_ollama(prompt)
        if not response_text:
            return candidates[:10]
        
        text = response_text.strip()
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        indices = json.loads(text.strip())
        
        reranked = []
        for idx in indices:
            if 0 <= idx < len(candidates):
                reranked.append(candidates[idx])
        
        seen = set(indices)
        for i, c in enumerate(candidates):
            if i not in seen:
                reranked.append(c)
        
        return reranked
        
    except Exception as e:
        logger.error(f"Ollama rerank failed: {e}")
        return candidates[:10]


def generate_final_with_ollama(query: str, candidates: List[Dict]) -> List[Dict]:
    """Generate final recommendations with Ollama - fallback when Gemini fails."""
    if not candidates or not OLLAMA_AVAILABLE:
        return basic_ranking(query, candidates, 10)
    
    assessment_info = []
    for i, a in enumerate(candidates[:15]):
        info = f"{i+1}. {a.get('name', 'Unknown')}\n   URL: {a.get('url', '')}\n   Type: {', '.join(a.get('test_types', []))}"
        assessment_info.append(info)
    
    assessment_text = "\n\n".join(assessment_info)
    
    prompt = f"""Recommend TOP 10 relevant assessments for:

QUERY: {query}

ASSESSMENTS:
{assessment_text}

Return JSON array only:
[
  {{"name": "Name", "url": "https://...", "reason": "Why match", "score": 0.95}},
  ...
]"""

    try:
        response_text = generate_with_ollama(prompt)
        if not response_text:
            return basic_ranking(query, candidates, 10)
        
        text = response_text.strip()
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        result = json.loads(text.strip())
        return result[:10]
        
    except Exception as e:
        logger.error(f"Ollama generation failed: {e}")
        return basic_ranking(query, candidates, 10)


# ==================== GEMINI RERANKING ====================

def rerank_with_gemini(query: str, candidates: List[Dict]) -> List[Dict]:
    """Gemini reranking - Stage 1: Gemini 2.0 -> Stage 2: Gemini 2.5 -> Stage 3: Ollama -> Stage 4: Basic"""
    if not candidates:
        return candidates[:10] if candidates else []
    
    if not GEMINI_AVAILABLE:
        logger.info("[STAGE 3] Using Ollama for rerank...")
        return rerank_with_ollama(query, candidates)
    
    # Build candidate info (common for all Gemini models)
    candidate_list = []
    for i, c in enumerate(candidates[:RERANK_CANDIDATES]):
        name = c.get('name', '')
        url = c.get('url', '')
        test_types = ', '.join(c.get('test_types', []))
        desc = c.get('description', '')[:150]
        
        info = f"""{i}. NAME: {name}
   URL: {url.split('/')[-1]}
   TYPES: {test_types}
   DESC: {desc}"""
        candidate_list.append(info)
    
    candidates_text = "\n\n".join(candidate_list)
    
    # ULTRA-EXPLICIT PROMPT: Gemini MUST return ALL candidates
    prompt = f"""INSTRUCTIONS: You are a ranker. Your ONLY job is to reorder, NEVER to remove items.

STRICT REQUIREMENT: You MUST output EXACTLY {RERANK_CANDIDATES} indices - no more, no less.
DO NOT filter. DO NOT drop. Only reorder.

Input has {RERANK_CANDIDATES} candidates (index 0 to {RERANK_CANDIDATES-1}).

QUERY: {query}

CANDIDATES:
{candidates_text}

OUTPUT INSTRUCTIONS:
- Output ONLY a JSON array with {RERANK_CANDIDATES} index numbers
- Example: [0,5,2,1,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
- The array must contain every number from 0 to {RERANK_CANDIDATES-1} exactly once
- Do NOT add any other text

Return the full array:"""

    # STAGE 1: Try Gemini Model 1 (gemini-2.0-flash-lite)
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=RERANK_MODEL_1,
            contents=prompt
        )
        
        text = response.text.strip()
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        indices = json.loads(text.strip())
        
        reranked = []
        for idx in indices:
            if 0 <= idx < len(candidates):
                reranked.append(candidates[idx])
        
        seen = set(indices)
        for i, c in enumerate(candidates):
            if i not in seen:
                reranked.append(c)
        
        logger.info("[STAGE 1] Rerank: Gemini model OK")
        return reranked
        
    except Exception as e:
        logger.warning(f"[STAGE 1] Gemini model failed: {str(e)[:50]}")
    
    # STAGE 2: Try Gemini Model 2 (gemini-2.5-flash-lite)
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=RERANK_MODEL_2,
            contents=prompt
        )
        
        text = response.text.strip()
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        indices = json.loads(text.strip())
        
        reranked = []
        for idx in indices:
            if 0 <= idx < len(candidates):
                reranked.append(candidates[idx])
        
        seen = set(indices)
        for i, c in enumerate(candidates):
            if i not in seen:
                reranked.append(c)
        
        logger.info("[STAGE 2] Rerank: Gemini model 2 OK")
        return reranked
        
    except Exception as e:
        logger.warning(f"[STAGE 2] Gemini model 2 failed: {str(e)[:50]}")
    
    # STAGE 3: Fallback to Ollama
    logger.info("[STAGE 3] Using Ollama for rerank...")
    return rerank_with_ollama(query, candidates)


# ==================== FINAL RECOMMENDATIONS ====================

def generate_with_gemini(query: str, candidates: List[Dict]) -> List[Dict]:
    """Generate final recommendations - Stage 1: Gemini 2.0 -> Stage 2: Gemini 2.5 -> Stage 3: Ollama -> Stage 4: Basic"""
    if not candidates:
        return basic_ranking(query, candidates, 10)
    
    if not GEMINI_AVAILABLE:
        logger.info("[STAGE 3] Gemini not available, using Ollama...")
        return generate_final_with_ollama(query, candidates)
    
    assessment_info = []
    for i, a in enumerate(candidates[:15]):
        info = f"{i+1}. {a.get('name', 'Unknown')}\n   URL: {a.get('url', '')}\n   Type: {', '.join(a.get('test_types', []))}"
        assessment_info.append(info)
    
    assessment_text = "\n\n".join(assessment_info)
    
    prompt = f"""TASK: Given the already-reranked candidate list below, output the TOP 10 with scores.

The candidates have already been reranked by relevance. Your job is to:
1. Take the FIRST 10 items from the reranked list
2. Assign them scores (0.0-1.0) based on relevance to the query
3. Keep them in the SAME ORDER - do NOT reorder

QUERY: {query}

ALREADY RERANKED CANDIDATES (use first 10 in this exact order):
{assessment_text}

IMPORTANT: Output exactly 10 items in the SAME ORDER as provided above. Just add scores and reasons.

Return JSON array with 10 items in exact same order:
[
  {{"name": "Name", "url": "...", "reason": "...", "score": 0.95}},
  ...
]"""

    # STAGE 1: Try Gemini Model 1 (gemini-2.0-flash-lite)
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL_1,
            contents=prompt
        )

        text = response.text.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        result = json.loads(text.strip())
        logger.info("[STAGE 1] Generate: Gemini model OK")
        return result[:10]

    except Exception as e:
        logger.warning(f"[STAGE 1] Gemini model failed: {str(e)[:50]}")

    # STAGE 2: Try Gemini Model 2 (gemini-2.5-flash-lite)
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL_2,
            contents=prompt
        )

        text = response.text.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        result = json.loads(text.strip())
        logger.info("[STAGE 2] Generate: Gemini model 2 OK")
        return result[:10]

    except Exception as e:
        logger.warning(f"[STAGE 2] Gemini model 2 failed: {str(e)[:50]}")

    # STAGE 3: Fallback to Ollama
    logger.info("[STAGE 3] Using Ollama for generation...")
    return generate_final_with_ollama(query, candidates)


def basic_ranking(query: str, assessments: List[Dict], top_k: int = 10) -> List[Dict]:
    """Basic keyword ranking fallback."""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored = []
    for a in assessments:
        score = 0.0
        name_lower = a.get('name', '').lower()
        desc_lower = a.get('description', '').lower()
        test_types = ' '.join(a.get('test_types', [])).lower()
        
        for word in query_words:
            if word in name_lower:
                score += KW_WEIGHT_NAME
            if word in desc_lower:
                score += KW_WEIGHT_DESC
            if word in test_types:
                score += KW_WEIGHT_TEST_TYPES
        
        scored.append((a, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for a, score in scored[:top_k]:
        results.append({
            "name": a.get("name", "Unknown"),
            "url": a.get("url", ""),
            "score": min(score / 100.0, 1.0),
            "reason": f"Matches {a.get('test_types', ['Assessment'])}"
        })
    
    return results


# ==================== MAIN API ====================

_assessments: Optional[List[Dict]] = None
_faiss_index: Optional[faiss.Index] = None


def initialize():
    """Initialize global resources."""
    global _assessments, _faiss_index
    
    logger.info("Initializing V9 Recommender...")
    
    _assessments = load_assessments()
    _faiss_index, _ = load_faiss_index()
    
    logger.info(f"Initialized with {len(_assessments)} assessments")


def recommend(query: str, top_k: int = 10, use_rerank: bool = True) -> List[Dict]:
    """Main recommendation function."""
    global _assessments, _faiss_index
    
    if _assessments is None or _faiss_index is None:
        initialize()
    
    # Hybrid retrieval
    hybrid_results = hybrid_retrieve(query, _assessments, _faiss_index, top_k=HYBRID_TOP_K)
    
    candidate_indices = [idx for idx, score in hybrid_results]
    candidates = [_assessments[idx] for idx in candidate_indices]
    
    logger.info(f"Pre-rerank: {[c.get('name', '')[:25] for c in candidates[:3]]}")
    
    # Try Gemini first (cloud-based, efficient), fallback to Ollama if fails
    gemini_success = False
    if use_rerank and GEMINI_AVAILABLE:
        try:
            candidates = rerank_with_gemini(query, candidates)
            logger.info(f"Post-rerank (Gemini): {[c.get('name', '')[:25] for c in candidates[:3]]}")
            gemini_success = True
        except Exception as e:
            logger.warning(f"Gemini rerank error: {e}")
    
    # Fallback to Ollama if Gemini failed or not available
    if not gemini_success and use_rerank and OLLAMA_AVAILABLE:
        try:
            candidates = rerank_with_ollama(query, candidates)
            logger.info(f"Post-rerank (Ollama fallback): {[c.get('name', '')[:25] for c in candidates[:3]]}")
        except Exception as e:
            logger.warning(f"Ollama rerank error: {e}")
    
    # Use reranked candidates with position-based scoring
    try:
        final_recs = []
        for i, c in enumerate(candidates[:top_k]):
            position_score = 1.0 - (i * 0.05)
            final_recs.append({
                "name": c.get("name", "Unknown"),
                "url": c.get("url", ""),
                "score": max(position_score, 0.5),
                "reason": f"Ranked position {i+1} - matches {c.get('test_types', ['Assessment'])}"
            })
        return final_recs
    except Exception as e:
        logger.warning(f"Final ranking error: {e}")
    
    return basic_ranking(query, candidates, top_k)


def normalize_url_key(url: str) -> str:
    """Normalize URL for matching."""
    url = url.rstrip('/')
    url = url.replace('/solutions/products/product-catalog/view/', '/products/product-catalog/view/')
    return url.split('/')[-1]


if __name__ == "__main__":
    initialize()
    test = "Java developer assessment"
    recs = recommend(test, top_k=5)
    for r in recs:
        print(f"- {r['name']}")
