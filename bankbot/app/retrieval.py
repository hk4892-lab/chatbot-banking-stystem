"""Lightweight retrieval augmented generation utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .language import detect_language, normalize_text

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - fallback path when sklearn missing
    TfidfVectorizer = None  # type: ignore[assignment]
    cosine_similarity = None  # type: ignore[assignment]


LOGGER = logging.getLogger("bankbot.retrieval")

DATA_DIR = Path(__file__).parent / "data"
KB_FILES = [
    DATA_DIR / "kb_en.json",
    DATA_DIR / "kb_ta.json",
    DATA_DIR / "kb_hi.json",
]

SCORE_THRESHOLD: float = 0.18

KB_ENTRIES: List[Dict[str, Any]] = []
KB_SIZE: int = 0
LANGS: List[str] = []

_TFIDF_VECTORIZER: Optional[TfidfVectorizer] = None
_TFIDF_MATRIX = None
_RETRIEVAL_READY = False


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_index() -> None:
    global _TFIDF_VECTORIZER, _TFIDF_MATRIX, _RETRIEVAL_READY

    if TfidfVectorizer is None:
        LOGGER.warning("scikit-learn not available; retrieval will be keyword-based only.")
        _RETRIEVAL_READY = bool(KB_ENTRIES)
        return

    corpus = [normalize_text(entry.get("q", "")) for entry in KB_ENTRIES]
    if not corpus:
        return

    _TFIDF_VECTORIZER = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    _TFIDF_MATRIX = _TFIDF_VECTORIZER.fit_transform(corpus)
    _RETRIEVAL_READY = True


def load_kb() -> List[Dict[str, Any]]:
    """Load the multilingual knowledge base on import."""

    if KB_ENTRIES:
        return KB_ENTRIES

    langs: Set[str] = set()

    for kb_file in KB_FILES:
        try:
            entries = _load_json(kb_file)
            KB_ENTRIES.extend(entries)
            langs.update(entry.get("lang", "en") for entry in entries)
        except FileNotFoundError:
            LOGGER.warning("Knowledge base file missing: %s", kb_file)
        except json.JSONDecodeError as exc:
            LOGGER.error("Failed to parse %s: %s", kb_file, exc)

    global KB_SIZE, LANGS
    KB_SIZE = len(KB_ENTRIES)
    LANGS = sorted(langs) if langs else ["en"]

    _build_index()
    return KB_ENTRIES


def _keyword_fallback(query: str) -> Dict[str, Any]:
    normalized_query = normalize_text(query)
    best_entry: Optional[Dict[str, Any]] = None
    best_score = 0
    for entry in KB_ENTRIES:
        normalized_q = normalize_text(entry.get("q", ""))
        if normalized_query and normalized_query in normalized_q:
            best_entry = entry
            best_score = 0.25
            break
    lang = detect_language(query)
    if not best_entry:
        return {"lang": lang, "score": 0.0, "answer": "", "intent": ""}
    return {
        "lang": best_entry.get("lang", lang),
        "score": best_score,
        "answer": best_entry.get("a", ""),
        "intent": best_entry.get("intent", ""),
    }


def retrieve(query: str) -> Dict[str, Any]:
    """Retrieve the top knowledge base answer for the given query."""

    if not KB_ENTRIES:
        load_kb()

    if not query.strip():
        lang = detect_language(query)
        return {"lang": lang, "score": 0.0, "answer": "", "intent": ""}

    lang = detect_language(query)

    if _TFIDF_VECTORIZER is None or _TFIDF_MATRIX is None:
        return _keyword_fallback(query)

    query_vector = _TFIDF_VECTORIZER.transform([normalize_text(query)])
    similarities = cosine_similarity(query_vector, _TFIDF_MATRIX)[0]
    best_index = int(similarities.argmax())
    best_score = float(similarities[best_index])
    best_entry = KB_ENTRIES[best_index]

    return {
        "lang": best_entry.get("lang", lang),
        "score": best_score,
        "answer": best_entry.get("a", ""),
        "intent": best_entry.get("intent", ""),
    }


def is_ready() -> bool:
    return _RETRIEVAL_READY


load_kb()


__all__ = [
    "retrieve",
    "load_kb",
    "is_ready",
    "KB_ENTRIES",
    "KB_SIZE",
    "LANGS",
    "SCORE_THRESHOLD",
]
