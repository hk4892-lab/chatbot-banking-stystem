"""FAISS-based retrieval with SentenceTransformers embeddings."""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FAISSRetriever:
    """FAISS retriever for semantic search over knowledge base."""

    def __init__(
        self,
        index_path: str,
        meta_path: str,
        embedding_model: str,
    ):
        """
        Initialize retriever with pre-built index.
        
        Args:
            index_path: Path to FAISS index file
            meta_path: Path to metadata pickle file
            embedding_model: HuggingFace model name for embeddings
        """
        self.embedding_model_name = embedding_model
        self.encoder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        
        # Lazy loading flags
        self._encoder_loaded = False
        self._index_loaded = False
        
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)

    def _load_encoder(self):
        """Lazy load the embedding encoder."""
        if not self._encoder_loaded:
            self.encoder = SentenceTransformer(self.embedding_model_name)
            self._encoder_loaded = True

    def _load_index(self):
        """Lazy load the FAISS index and metadata."""
        if not self._index_loaded:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
            
            if self.meta_path.exists():
                with open(self.meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
            
            self._index_loaded = True

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for top-k relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of dicts with score, doc_id, title, content, lang, etc.
        """
        self._load_encoder()
        self._load_index()
        
        if self.index is None or not self.metadata:
            return []
        
        # Encode query and normalize
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)
        query_vector = query_embedding.astype("float32")
        
        # Search FAISS
        k_actual = min(k, len(self.metadata))
        scores, indices = self.index.search(query_vector, k_actual)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(score)
                results.append(result)
        
        return results


def build_index(
    kb_chunks_path: str,
    output_index_path: str,
    output_meta_path: str,
    embedding_model: str,
) -> None:
    """
    Build FAISS index from KB chunks.
    
    Args:
        kb_chunks_path: Path to JSONL file with KB chunks
        output_index_path: Where to save FAISS index
        output_meta_path: Where to save metadata pickle
        embedding_model: HuggingFace model for embeddings
    """
    # Load encoder
    encoder = SentenceTransformer(embedding_model)
    
    # Load KB chunks
    chunks = []
    with open(kb_chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    if not chunks:
        raise ValueError(f"No chunks found in {kb_chunks_path}")
    
    # Extract content for embedding
    texts = [chunk["content"] for chunk in chunks]
    
    # Generate embeddings
    print(f"Encoding {len(texts)} chunks...")
    embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")
    
    # Build FAISS index (Inner Product for cosine similarity with normalized vectors)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, output_index_path)
    print(f"Saved FAISS index to {output_index_path}")
    
    # Save metadata
    with open(output_meta_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved metadata to {output_meta_path}")
