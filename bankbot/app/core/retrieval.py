"""Retrieval module supporting TF-IDF and optional sentence-transformers."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .language import normalize_text

try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class KBItem:
    id: str
    title: str
    content: str
    tags: List[str]
    lang: str


@dataclass
class SearchResult:
    item: KBItem
    score: float


@dataclass
class SearchResponse:
    results: List[SearchResult]

    @property
    def top_score(self) -> float:
        return self.results[0].score if self.results else 0.0

    @property
    def top_item(self) -> Optional[KBItem]:
        return self.results[0].item if self.results else None


class Retriever:
    """TF-IDF based retriever with optional sentence-transformer reranker."""

    def __init__(self, data_dir: Path, threshold: float = 0.18) -> None:
        self.data_dir = data_dir
        self.threshold = threshold
        self.items: List[KBItem] = []
        self._load_items()

        self.vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1, 2))
        corpus = [normalize_text(item.content) for item in self.items]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        self._sentence_model: Optional[SentenceTransformer] = None
        self._sentence_embeddings: Optional[np.ndarray] = None

    def _load_items(self) -> None:
        for lang_code in ("en", "hi", "ta"):
            path = self.data_dir / f"kb_{lang_code}.json"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                entries = json.load(f)
            for entry in entries:
                item = KBItem(
                    id=entry["id"],
                    title=entry["title"],
                    content=entry["content"],
                    tags=entry.get("tags", []),
                    lang=lang_code.upper(),
                )
                self.items.append(item)

    def _ensure_sentence_model(self) -> bool:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return False
        if self._sentence_model is None:
            # Lazy load to keep startup fast
            self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._sentence_embeddings = self._sentence_model.encode(
                [normalize_text(item.content) for item in self.items], show_progress_bar=False
            )
        return True

    def search(
        self,
        query: str,
        lang_hint: str,
        top_k: int = 3,
        use_sentence_transformers: bool = False,
    ) -> SearchResponse:
        if not query.strip():
            return SearchResponse(results=[])

        normalized_query = normalize_text(query)
        query_vec = self.vectorizer.transform([normalized_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        lang_hint = lang_hint.upper()
        # Boost same-language documents slightly for tie-breaking
        for idx, item in enumerate(self.items):
            if item.lang == lang_hint:
                scores[idx] += 0.01

        top_indices = np.argsort(scores)[::-1][:max(top_k, 3)]

        results: List[SearchResult] = [
            SearchResult(item=self.items[idx], score=float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

        if use_sentence_transformers and self._ensure_sentence_model():
            assert self._sentence_model is not None
            assert self._sentence_embeddings is not None
            query_embedding = self._sentence_model.encode(normalized_query, show_progress_bar=False)
            st_scores = cosine_similarity([query_embedding], self._sentence_embeddings)[0]
            for res in results:
                item_index = self.items.index(res.item)
                res.score = float((res.score + st_scores[item_index]) / 2)
            results.sort(key=lambda r: r.score, reverse=True)

        return SearchResponse(results=results[:top_k])


def format_citations(results: Sequence[SearchResult]) -> List[Dict[str, str]]:
    """Create citation objects for API responses."""

    citations: List[Dict[str, str]] = []
    for res in results:
        citations.append({"id": res.item.id, "title": res.item.title})
    return citations
