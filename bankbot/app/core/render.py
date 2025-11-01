"""Utilities for formatting chatbot responses with citations."""
from __future__ import annotations

from typing import Sequence

from .retrieval import SearchResult


def render_answer(body: str, citations: Sequence[SearchResult]) -> str:
    """Append citations to the answer body."""

    if not citations:
        return body
    citation_text = ", ".join(f"{res.item.title} ({res.item.id})" for res in citations)
    return f"{body}\n\nSources: {citation_text}"
