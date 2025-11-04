"""Vector store utilities backed by FAISS with NumPy fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .embeddings import embed_passages

try:  # pragma: no cover - optional dependency path
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover - fallback path
    faiss = None  # type: ignore
    _HAS_FAISS = False


class VectorStore:
    """Simple wrapper around FAISS or NumPy cosine search."""

    def __init__(self) -> None:
        self._use_faiss = _HAS_FAISS
        self._index = None
        self._vectors: np.ndarray | None = None
        self._passages: List[str] = []
        self._metas: List[Dict] = []

    @property
    def is_ready(self) -> bool:
        if self._use_faiss:
            return self._index is not None and bool(self._passages)
        return self._vectors is not None and bool(self._passages)

    def build(self, passages: Sequence[str], metas: Sequence[Dict]) -> None:
        if len(passages) != len(metas):
            raise ValueError("Passages and metas must have the same length")

        vectors = embed_passages(passages)
        self._passages = list(passages)
        self._metas = [dict(meta) for meta in metas]

        if self._use_faiss:
            dimension = vectors.shape[1]
            self._index = faiss.IndexFlatIP(dimension)  # type: ignore[attr-defined]
            self._index.add(vectors)
        else:
            self._vectors = vectors

    def save(self, directory: str) -> None:
        if not self.is_ready:
            raise RuntimeError("Vector store is not built")

        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        payload = {"passages": self._passages, "metas": self._metas, "use_faiss": self._use_faiss}
        with (dir_path / "store_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        if self._use_faiss:
            if self._index is None:
                raise RuntimeError("FAISS index not initialized")
            faiss.write_index(self._index, str(dir_path / "index.faiss"))  # type: ignore[attr-defined]
        else:
            if self._vectors is None:
                raise RuntimeError("Vector matrix not initialized")
            np.save(dir_path / "vectors.npy", self._vectors)

    def load(self, directory: str) -> None:
        dir_path = Path(directory)
        meta_path = dir_path / "store_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)

        with meta_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self._passages = list(payload.get("passages", []))
        self._metas = list(payload.get("metas", []))
        self._use_faiss = bool(payload.get("use_faiss", _HAS_FAISS)) and _HAS_FAISS

        if self._use_faiss:
            index_path = dir_path / "index.faiss"
            if not index_path.exists():
                raise FileNotFoundError(index_path)
            self._index = faiss.read_index(str(index_path))  # type: ignore[attr-defined]
            self._vectors = None
        else:
            vector_path = dir_path / "vectors.npy"
            if not vector_path.exists():
                raise FileNotFoundError(vector_path)
            self._vectors = np.load(vector_path)
            self._index = None

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        if not self.is_ready:
            raise RuntimeError("Vector store not built or loaded")

        top_k = max(1, top_k)

        if self._use_faiss:
            assert self._index is not None
            distances, indices = self._index.search(query_vector.reshape(1, -1), top_k)  # type: ignore[attr-defined]
            scores = distances[0]
            idxs = indices[0]
        else:
            assert self._vectors is not None
            scores = self._vectors @ query_vector.reshape(-1, 1)
            scores = scores.ravel()
            idxs = scores.argsort()[::-1][:top_k]
            scores = scores[idxs]

        results: List[Dict] = []
        for score, idx in zip(scores, idxs):
            if idx < 0 or idx >= len(self._passages):
                continue
            results.append(
                {
                    "score": float(score),
                    "text": self._passages[idx],
                    "meta": self._metas[idx],
                }
            )
        return results


__all__ = ["VectorStore"]
