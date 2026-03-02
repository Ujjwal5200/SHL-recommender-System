"""
SHL Assessment Recommender - Smart Hybrid without LLM

Uses better keyword matching and semantic search
"""

import json
import re
from typing import List, Dict, Any
from collections import defaultdict

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from src.config import Config
from src.logger import logger


def load_assessments() -> List[Dict[str, Any]]:
    with open(Config.ASSESSMENTS_JSON, 'r', encoding='utf-8') as f:
        assessments = json.load(f)
    for i, a in enumerate(assessments):
        a['idx'] = i
    return assessments


def load_vectorstore():
    embeddings = OllamaEmbeddings(
        model=Config.EMBEDDING_MODEL,
        base_url=Config.OLLAMA_BASE_URL
    )
    return FAISS.load_local(
        Config.INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def extract_keywords(query: str) -> set:
    """Extract keywords from query."""
    query = query.lower()
    
    # Comprehensive keywords
    keywords = {
        # Programming
        'java', 'python', 'sql', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust',
        'php', 'swift', 'kotlin', 'scala', 'r', 'perl', 'typescript', 'html', 'css',
        'xml', 'json', 'sql server',
        # Data
        'excel', 'tableau', 'powerbi', 'bi', 'analytics', 'data', 'statistics',
        'hadoop', 'spark', 'aws', 'azure', 'gcp', 'cloud', 'database',
        'mysql', 'postgresql', 'oracle', 'mongodb', 'nosql',
        # Tools
        'selenium', 'jira', 'git', 'docker', 'kubernetes', 'jenkins',
        'spring', 'django', 'react', 'angular', 'vue', 'node', 'jquery',
        # Jobs
        'developer', 'engineer', 'analyst', 'manager', 'designer', 'consultant',
        'architect', 'senior', 'junior', 'intern',
        'qa', 'tester', 'admin', 'administrator', 'sales', 'marketing',
        'accountant', 'hr', 'recruiter', 'writer', 'editor', 'specialist',
        'assistant', 'coordinator', 'director', 'supervisor', 'coo', 'ceo', 'cto',
        'administrative', 'professional', 'executive', 'entry', 'graduate',
        # Skills
        'communication', 'leadership', 'teamwork', 'problem solving', 'analytical',
        'programming', 'coding', 'testing', 'security', 'networking',
        'project management', 'agile', 'cognitive', 'aptitude',
        'collaboration', 'interpersonal', 'presentation', 'negotiation',
        'time management', 'adaptability', 'creativity', 'verbal', 'numerical',
        'personality', 'behavior', 'cultural', 'fit',
        # Other
        'quality assurance', 'test automation', 'manual testing',
        'content', 'english', 'seo', 'writing', 'digital marketing',
        'customer service', 'retail', 'banking', 'finance',
        'human resources', 'recruitment', 'talent',
        'operations', 'supply chain', 'logistics',
        'data science', 'machine learning', 'ai', 'artificial intelligence',
    }
    
    found = set()
    words = re.findall(r'\b\w+\b', query)
    for word in words:
        if word in keywords:
            found.add(word)
    
    # Phrases
    phrases = [
        'problem solving', 'project management', 'time management',
        'data analyst', 'software developer', 'quality assurance',
        'test automation', 'business analyst', 'product manager',
        'customer service', 'content writer', 'digital marketing',
        'machine learning', 'artificial intelligence', 'deep learning',
        'entry level', 'new graduate', 'senior analyst',
    ]
    
    for phrase in phrases:
        if phrase in query:
            found.add(phrase)
    
    return found


def keyword_score(query: str, assessments: List[Dict[str, Any]]) -> Dict[int, float]:
    """Score assessments by keyword matching."""
    keywords = extract_keywords(query)
    scores = defaultdict(float)
    
    for a in assessments:
        name = a.get('name', '').lower()
        desc = a.get('description', '').lower()
        test_types = ' '.join(a.get('test_types', [])).lower()
        url = a.get('url', '').lower()
        
        # URL slug
        slug = url.split('/view/')[-1].rstrip('/') if '/view/' in url else url
        
        combined = f"{name} {desc} {test_types} {slug}"
        
        for kw in keywords:
            # Priority: slug > name > test_types > description
            if kw in slug:
                scores[a['idx']] += 5.0
            if kw in name:
                scores[a['idx']] += 4.0
            if kw in test_types:
                scores[a['idx']] += 2.0
            if kw in desc:
                scores[a['idx']] += 1.0
            if len(kw) > 4 and kw in combined:
                scores[a['idx']] += 0.5
    
    return dict(scores)


def semantic_score(query: str, k: int = 30) -> Dict[int, float]:
    """Get semantic similarity scores from FAISS."""
    try:
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)

        scores = {}
        assessments = load_assessments()
        for i, d in enumerate(docs):
            name = d.metadata.get("name", "")
            for a in assessments:
                if a.get("name") == name:
                    scores[a['idx']] = 1.0 - (i * 0.02)
                    break
        return scores
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return {}


def get_recommendations(query: str, top_n: int = 10) -> List[Dict[str, Any]]:
    """Main recommendation function."""
    top_n = max(5, min(10, top_n))
    
    assessments = load_assessments()
    
    # Get keyword scores
    kw_scores = keyword_score(query, assessments)
    logger.info(f"Keyword matches: {len(kw_scores)}")
    
    # Get semantic scores
    sem_scores = semantic_score(query, k=30)
    logger.info(f"Semantic matches: {len(sem_scores)}")
    
    # Combine scores
    combined = {}
    for idx, score in kw_scores.items():
        combined[idx] = {'keyword': score * 2.0, 'semantic': 0, 'total': score * 2.0}
    
    for idx, score in sem_scores.items():
        if idx in combined:
            combined[idx]['semantic'] = score
            combined[idx]['total'] += score
        else:
            combined[idx] = {'keyword': 0, 'semantic': score, 'total': score}
    
    # Sort by total score
    sorted_results = sorted(combined.items(), key=lambda x: x[1]['total'], reverse=True)
    
    # Get top results
    results = []
    for idx, _ in sorted_results[:top_n]:
        a = assessments[idx]
        results.append({
            "name": a.get("name", ""),
            "url": a.get("url", ""),
            "test_types": a.get("test_types", []),
            "duration_minutes": a.get("duration_minutes", 0),
            "adaptive_support": a.get("adaptive_support", "No"),
            "remote_support": a.get("remote_support", "No"),
            "description": a.get("description", "")[:500],
        })
    
    # Ensure at least 5
    while len(results) < 5:
        results.append(results[0] if results else {
            "name": "N/A", "url": "", "test_types": [],
            "duration_minutes": 0, "adaptive_support": "No",
            "remote_support": "No", "description": ""
        })
    
    return results[:10]
