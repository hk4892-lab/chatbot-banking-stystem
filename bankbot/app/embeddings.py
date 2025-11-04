"""Embedding utilities using sentence-transformers."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import numpy as np

from .config import EMBED_MODEL


class _Embedder:
    """Singleton wrapper around SentenceTransformer with normalization."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.is_e5 = model_name.lower().startswith("intfloat/multilingual-e5")

    def _format_queries(self, queries: Iterable[str]) -> List[str]:
        if not self.is_e5:
            return list(queries)
        return [f"query: {text}" for text in queries]

    def _format_passages(self, passages: Iterable[str]) -> List[str]:
        if not self.is_e5:
            return list(passages)
        return [f"passage: {text}" for text in passages]

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def encode_queries(self, queries: Iterable[str]) -> np.ndarray:
        return self.encode(self._format_queries(list(queries)))

    def encode_passages(self, passages: Iterable[str]) -> np.ndarray:
        return self.encode(self._format_passages(list(passages)))


@lru_cache(maxsize=1)
def get_embedder() -> _Embedder:
    return _Embedder(EMBED_MODEL)


def embed_queries(texts: Iterable[str]) -> np.ndarray:
    return get_embedder().encode_queries(list(texts))


def embed_passages(texts: Iterable[str]) -> np.ndarray:
    return get_embedder().encode_passages(list(texts))


__all__ = ["get_embedder", "embed_queries", "embed_passages"]
