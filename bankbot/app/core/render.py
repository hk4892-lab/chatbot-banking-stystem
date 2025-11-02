"""Response rendering helpers."""

from __future__ import annotations

from typing import Sequence

from .retrieval import RetrievalResult


def citations_from_results(results: Sequence[RetrievalResult]) -> list[dict[str, str]]:
    """Format citations from retrieval results."""

    return [
        {"id": item.entry.id, "title": item.entry.title}
        for item in list(results)[:3]
    ]


def answer_from_results(results: Sequence[RetrievalResult]) -> str:
    """Pick the best answer text from retrieval output."""

    if not results:
        return ""
    best = results[0].entry
    return best.content


def clarifying_prompt(result: RetrievalResult) -> tuple[str, list[str]]:
    """Generate a clarifying question and two example rephrases."""

    topic = result.entry.title
    question = f"Could you clarify your question about {topic.lower()}?"
    suggestions: list[str] = []

    if result.entry.tags:
        suggestions.append(f"Tell me if this is about {result.entry.tags[0].lower()}.")
    suggestions.append("Share any specific account or timeline details without sensitive numbers.")

    return question, suggestions[:2]
