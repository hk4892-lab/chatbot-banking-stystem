from __future__ import annotations

from pathlib import Path

from app.core.retrieval import RetrievalEngine


DATA_DIR = Path(__file__).resolve().parents[1] / "app" / "data"


def test_hinglish_query_returns_upi_entry() -> None:
    retriever = RetrievalEngine(DATA_DIR, use_sentence_transformers=False)
    results = retriever.search("UPI limit kitna hai", top_k=1)

    assert results, "Expected at least one retrieval result"
    top = results[0]
    assert top.score > 0.18
    assert "UPI" in top.entry.title.upper()
