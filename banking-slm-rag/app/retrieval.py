"""FAISS-backed retrieval utilities."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings


@dataclass
class KnowledgeChunk:
    doc_id: str
    title: str
    content: str
    lang: str
    tags: Sequence[str]
    updated_on: str


@dataclass
class RetrievalResult:
    chunk: KnowledgeChunk
    score: float


def _load_chunks(path: Path) -> List[KnowledgeChunk]:
    chunks: List[KnowledgeChunk] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            chunks.append(
                KnowledgeChunk(
                    doc_id=payload['doc_id'],
                    title=payload['title'],
                    content=payload['content'],
                    lang=payload.get('lang', 'EN'),
                    tags=payload.get('tags', []),
                    updated_on=payload.get('updated_on', ''),
                )
            )
    return chunks


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


class FaissRetriever:
    """Wrapper over FAISS inner-product index for cosine similarity retrieval."""

    def __init__(
        self,
        settings: Settings,
        embedding_model: SentenceTransformer,
        *,
        auto_load: bool = True,
    ) -> None:
        self.settings = settings
        self.embedding_model = embedding_model
        self.index_path = settings.faiss_index_path
        self.meta_path = settings.faiss_meta_path
        self.source_path = settings.kb_source_path
        self.index: faiss.Index | None = None
        self.meta: List[KnowledgeChunk] = []
        if auto_load:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with self.meta_path.open('rb') as handle:
                    self.meta = pickle.load(handle)
                return
            except (OSError, ValueError, pickle.UnpicklingError):
                # Fallback to rebuild.
                pass
        self._build_from_source()

    def _build_from_source(self) -> None:
        if not self.source_path.exists():
            raise FileNotFoundError(f'Knowledge base source file missing at {self.source_path}')
        chunks = _load_chunks(self.source_path)
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype(np.float32))
        self.index = index
        self.meta = chunks
        self._persist()

    def _persist(self) -> None:
        if self.index is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open('wb') as handle:
            pickle.dump(self.meta, handle)

    def search(self, query: str, k: int) -> List[RetrievalResult]:
        if self.index is None:
            raise RuntimeError('Retriever index unavailable')
        k = min(k, len(self.meta))
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = _normalize(query_embedding.astype(np.float32))
        scores, indices = self.index.search(query_embedding, k)
        top_scores = scores[0]
        top_indices = indices[0]
        results: List[RetrievalResult] = []
        for score, idx in zip(top_scores, top_indices):
            if idx == -1:
                continue
            chunk = self.meta[idx]
            results.append(RetrievalResult(chunk=chunk, score=float(score)))
        return results


def build_index(
    settings: Settings,
    embedding_model: SentenceTransformer,
    chunks: Iterable[KnowledgeChunk],
    persist: bool = True,
) -> FaissRetriever:
    """Construct a new index from provided chunks."""

    texts = [chunk.content for chunk in chunks]
    vectors = embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors.astype(np.float32))
    retriever = FaissRetriever(settings=settings, embedding_model=embedding_model, auto_load=False)
    retriever.index = index
    retriever.meta = list(chunks)
    if persist:
        retriever._persist()
    return retriever


__all__ = ['FaissRetriever', 'KnowledgeChunk', 'RetrievalResult', 'build_index']
