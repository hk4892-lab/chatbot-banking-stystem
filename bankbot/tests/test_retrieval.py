"""Tests for retrieval behavior."""
from __future__ import annotations

from pathlib import Path

from app.core.retrieval import Retriever


def test_hinglish_query_hits_upi_entry() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "app" / "data"
    retriever = Retriever(data_dir)

    response = retriever.search("UPI limit kitna hai", lang_hint="HI", top_k=3)

    assert response.results
    assert response.top_score > 0.18
    assert response.top_item is not None
    assert "UPI" in response.top_item.title.upper()
