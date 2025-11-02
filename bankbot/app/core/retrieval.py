"""Retrieval utilities for the hybrid RAG setup."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .language import Language, normalize_text

LOGGER = logging.getLogger(__name__)


@dataclass
class KBEntry:
    """Structured knowledge base entry."""

    id: str
    title: str
    content: str
    tags: Sequence[str]
    lang: Language


@dataclass
class RetrievalResult:
    """Result returned by the retriever."""

    entry: KBEntry
    score: float


MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class RetrievalEngine:
    """TF-IDF baseline with optional SentenceTransformer re-ranking."""

    def __init__(self, data_dir: Path, use_sentence_transformers: bool = False) -> None:
        self.entries = _load_entries(data_dir)
        if not self.entries:
            raise ValueError("Knowledge base is empty")

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False)
        corpus = [self._entry_to_text(entry) for entry in self.entries]
        self.matrix = self.vectorizer.fit_transform(corpus)

        self._encoder = None
        self._doc_embeddings: np.ndarray | None = None
        if use_sentence_transformers:
            self._maybe_boot_encoder(corpus)

    def _entry_to_text(self, entry: KBEntry) -> str:
        tags = " ".join(entry.tags)
        return normalize_text(" ".join([entry.title, entry.content, tags]))

    def _maybe_boot_encoder(self, corpus: list[str]) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(MULTILINGUAL_MODEL)
            self._doc_embeddings = self._encoder.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("SentenceTransformer unavailable: %s", exc)
            self._encoder = None
            self._doc_embeddings = None

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        """Return top-k documents for the given query."""

        normalized_query = normalize_text(query)
        if not normalized_query:
            return []

        query_vec = self.vectorizer.transform([normalized_query])
        scores = cosine_similarity(query_vec, self.matrix)[0]

        if self._encoder and self._doc_embeddings is not None:
            embedding = self._encoder.encode([normalized_query], convert_to_numpy=True, normalize_embeddings=True)[
                0
            ]
            st_scores = np.dot(self._doc_embeddings, embedding)
            scores = (scores + st_scores) / 2

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(RetrievalResult(entry=self.entries[int(idx)], score=score))

        return results


def _load_entries(data_dir: Path) -> list[KBEntry]:
    entries: list[KBEntry] = []
    for lang_code, filename in (
        (Language.EN, "kb_en.json"),
        (Language.HI, "kb_hi.json"),
        (Language.TA, "kb_ta.json"),
    ):
        path = data_dir / filename
        if not path.exists():
            LOGGER.warning("KB file missing: %s", path)
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for item in payload:
            entries.append(
                KBEntry(
                    id=item["id"],
                    title=item["title"],
                    content=item["content"],
                    tags=item.get("tags", []),
                    lang=lang_code,
                )
            )
    return entries
