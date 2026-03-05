# src/indexer.py
"""
FAISS Index Builder - V6 Hybrid RAG System
Builds FAISS index using sentence-transformers for semantic search.
"""
import json
import numpy as np
import faiss
import logging
from config_v9 import DATA_DIR, FAISS_DIR, EMBED_MODEL, logger
from tqdm.auto import tqdm
from pathlib import Path
from glob import glob

from config_v9 import DATA_DIR, FAISS_DIR, EMBED_MODEL, logger


def get_latest_catalog() -> Path:
    """Find the latest SHL catalog JSON file."""
    files = sorted(glob(str(DATA_DIR / "shl_individual_tests_*.json")), reverse=True)
    if not files:
        raise FileNotFoundError("No catalog json found in data/")
    return Path(files[0])


def build_faiss_index(embed_model: str = None):
    """
    Build FAISS index from SHL catalog using sentence-transformers.
    
    Args:
        embed_model: Model name for sentence-transformers (default from config)
    """
    if embed_model is None:
        embed_model = EMBED_MODEL
    
    # Try to use sentence-transformers first
    use_sentence_transformers = False
    try:
        from sentence_transformers import SentenceTransformer
        use_sentence_transformers = True
        logger.info(f"Using sentence-transformers: {embed_model}")
    except ImportError:
        logger.warning("sentence-transformers not available, using Ollama")
    
    # Use merged metadata.json if it exists, otherwise use latest catalog
    meta_path = FAISS_DIR / "metadata.json"
    if meta_path.exists():
        catalog_path = meta_path
        logger.info(f"Using merged metadata: {catalog_path}")
    else:
        catalog_path = get_latest_catalog()
        logger.info(f"Using catalog: {catalog_path}")

    documents = []
    embed_texts = []

    with open(catalog_path, encoding="utf-8") as f:
        items = json.load(f)
        logger.info(f"Loaded {len(items)} items from {catalog_path}")
        for item in items:
            documents.append(item)
            # Create rich text for embedding
            text = f"{item['name']} | {', '.join(item.get('test_types', []))} | {item.get('description', '')[:450]}"
            embed_texts.append(text)

    logger.info(f"Generating embeddings for {len(embed_texts)} items ...")

    if use_sentence_transformers:
        # Use sentence-transformers
        model = SentenceTransformer(embed_model)
        embeddings = model.encode(embed_texts, show_progress_bar=True, convert_to_numpy=True)
    else:
        # Fallback to Ollama
        import ollama
        embeddings = []
        for txt in tqdm(embed_texts, desc="Embedding (Ollama)"):
            try:
                resp = ollama.embeddings(model="nomic-embed-text", prompt=txt)
                emb = np.array(resp["embedding"], dtype=np.float32)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                continue
        
        if not embeddings:
            raise ValueError("No embeddings generated")
        
        embeddings = np.array(embeddings)

    emb_array = np.array(embeddings)
    d = emb_array.shape[1]

    # CRITICAL: Normalize the entire array before adding to index
    faiss.normalize_L2(emb_array)

    index = faiss.IndexFlatIP(d)  # Inner Product = cosine after normalization
    index.add(emb_array)

    FAISS_DIR.mkdir(exist_ok=True, parents=True)
    faiss.write_index(index, str(FAISS_DIR / "index.faiss"))

    meta_path = FAISS_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    logger.info(f"Index built — {len(documents)} items, dim {d}")
    logger.info(f"Metadata saved: {meta_path}")
    
    return index


def load_index():
    """Load existing FAISS index and metadata."""
    index_path = FAISS_DIR / "index.faiss"
    meta_path = FAISS_DIR / "metadata.json"
    
    if not index_path.exists() or not meta_path.exists():
        logger.warning("Index not found, building new one...")
        build_faiss_index()
    
    index = faiss.read_index(str(index_path))
    
    with open(meta_path, encoding="utf-8") as f:
        documents = json.load(f)
    
    logger.info(f"Loaded index with {index.ntotal} vectors, {len(documents)} documents")
    return index, documents


if __name__ == "__main__":
    build_faiss_index()

