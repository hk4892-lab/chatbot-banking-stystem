"""Lightweight language utilities for BankBot."""

from __future__ import annotations

import re
import string


_TAMIL_RANGE = re.compile(r"[\u0B80-\u0BFF]")
_DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
_PUNCT_TRANS = str.maketrans({ch: " " for ch in string.punctuation})


def detect_language(text: str | None) -> str:
    """Detect Tamil, Hindi, or default to English based on Unicode ranges."""

    if not text:
        return "en"
    if _TAMIL_RANGE.search(text):
        return "ta"
    if _DEVANAGARI_RANGE.search(text):
        return "hi"
    return "en"


def normalize(text: str | None) -> str:
    """Normalize text for retrieval: lowercase, strip punctuation, collapse spaces."""

    if not text:
        return ""
    lowered = text.lower().translate(_PUNCT_TRANS)
    compact = re.sub(r"\s+", " ", lowered)
    return compact.strip()


__all__ = ["detect_language", "normalize"]
