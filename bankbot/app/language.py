"""Language detection and normalization utilities."""

from __future__ import annotations

import re
import string


_TAMIL_RANGE = re.compile(r"[\u0B80-\u0BFF]")
_DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
_PUNCT_TABLE = str.maketrans({char: " " for char in string.punctuation})


def detect_language(text: str | None) -> str:
    """Detect whether the text is Tamil, Hindi (Devanagari), or default to English."""

    if not text:
        return "en"

    if _TAMIL_RANGE.search(text):
        return "ta"
    if _DEVANAGARI_RANGE.search(text):
        return "hi"
    return "en"


def normalize_text(text: str | None) -> str:
    """Lowercase and strip punctuation for retrieval purposes."""

    if not text:
        return ""
    lowered = text.lower().translate(_PUNCT_TABLE)
    compact = re.sub(r"\s+", " ", lowered)
    return compact.strip()


__all__ = ["detect_language", "normalize_text"]
