"""Retrieval components combining embeddings and the vector store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .embeddings import embed_queries
from .vector_store import VectorStore


DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = DATA_DIR / "index"

_KB_CACHE: List[Dict] | None = None
_VECTOR_STORE = VectorStore()
_INDEX_READY = False


def load_kb() -> List[Dict]:
    global _KB_CACHE
    if _KB_CACHE is not None:
        return _KB_CACHE

    kb_items: List[Dict] = []
    for kb_path in sorted(DATA_DIR.glob("kb_*.json")):
        with kb_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        for item in records:
            question = item.get("q", "").strip()
            answer = item.get("a", "").strip()
            passage = f"{question} {answer}".strip()
            kb_items.append(
                {
                    "question": question,
                    "answer": answer,
                    "intent": item.get("intent", ""),
                    "lang": item.get("lang", "en"),
                    "passage": passage,
                }
            )
    _KB_CACHE = kb_items
    return _KB_CACHE


def _index_paths() -> Dict[str, Path]:
    return {
        "meta": INDEX_DIR / "store_meta.json",
        "faiss": INDEX_DIR / "index.faiss",
        "vectors": INDEX_DIR / "vectors.npy",
    }


def build_or_load_index() -> VectorStore:
    global _INDEX_READY

    paths = _index_paths()
    try:
        if paths["meta"].exists():
            _VECTOR_STORE.load(str(INDEX_DIR))
            _INDEX_READY = _VECTOR_STORE.is_ready
    except FileNotFoundError:
        _INDEX_READY = False

    if not _INDEX_READY:
        kb_items = load_kb()
        passages = [item["passage"] for item in kb_items]
        metas = [
            {
                "intent": item["intent"],
                "lang": item["lang"],
                "answer": item["answer"],
                "question": item["question"],
            }
            for item in kb_items
        ]
        _VECTOR_STORE.build(passages, metas)
        _VECTOR_STORE.save(str(INDEX_DIR))
        _INDEX_READY = True

    return _VECTOR_STORE


def retrieve(query: str, top_k: int) -> List[Dict]:
    store = build_or_load_index()
    if not query.strip():
        return []

    query_vec = embed_queries([query])[0]
    hits = store.search(query_vec, top_k)

    results: List[Dict] = []
    for hit in hits:
        meta = hit.get("meta", {})
        results.append(
            {
                "score": float(hit.get("score", 0.0)),
                "text": meta.get("answer") or hit.get("text", ""),
                "intent": meta.get("intent", ""),
                "lang": meta.get("lang", "en"),
                "question": meta.get("question", ""),
                "context": hit.get("text", ""),
            }
        )
    return results


def index_ready() -> bool:
    return _INDEX_READY and _VECTOR_STORE.is_ready


__all__ = ["load_kb", "build_or_load_index", "retrieve", "index_ready"]
