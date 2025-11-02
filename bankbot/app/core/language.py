"""Language detection and normalization utilities."""

from __future__ import annotations

import re
from enum import Enum


DEVANAGARI_RANGE = (0x0900, 0x097F)
TAMIL_RANGE = (0x0B80, 0x0BFF)


class Language(str, Enum):
    """Supported language codes."""

    EN = "EN"
    HI = "HI"
    TA = "TA"
    AUTO = "AUTO"


LATIN_PATTERN = re.compile(r"[A-Za-z]")
WHITESPACE_RE = re.compile(r"\s+")

SYNONYM_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\blimit kitna\b", re.IGNORECASE), "upi limit"),
    (re.compile(r"\bupi limit kitna\b", re.IGNORECASE), "upi limit"),
    (re.compile(r"\bemi calc karna\b", re.IGNORECASE), "emi calculate"),
    (re.compile(r"\bemi calculator\b", re.IGNORECASE), "emi calculate"),
    (re.compile(r"\bstatement send pannunga\b", re.IGNORECASE), "statement request"),
    (re.compile(r"\bstatement anuppunga\b", re.IGNORECASE), "statement request"),
    (re.compile(r"\bbalance kitna\b", re.IGNORECASE), "balance enquiry"),
    (re.compile(r"\bblock card karo\b", re.IGNORECASE), "block card"),
    (re.compile(r"\bcard block pannunga\b", re.IGNORECASE), "block card"),
)


def detect_language(text: str) -> Language:
    """Detect the dominant script in the text.

    Args:
        text: User provided text.

    Returns:
        The inferred language code based on script usage.
    """

    devanagari = _count_chars_in_range(text, DEVANAGARI_RANGE)
    tamil = _count_chars_in_range(text, TAMIL_RANGE)

    if devanagari > tamil and devanagari > 0:
        return Language.HI
    if tamil > 0:
        return Language.TA
    return Language.EN


def normalize_text(text: str) -> str:
    """Normalize user input for retrieval.

    - Collapses whitespace
    - Lowercases Latin script tokens
    - Applies simple synonym rewrites for Hinglish/Tanglish tokens

    Args:
        text: Raw user text.

    Returns:
        Normalized text suitable for retrieval.
    """

    collapsed = WHITESPACE_RE.sub(" ", text).strip()
    if not collapsed:
        return ""

    tokens: list[str] = []
    for token in collapsed.split(" "):
        if LATIN_PATTERN.search(token):
            tokens.append(token.lower())
        else:
            tokens.append(token)

    normalized = " ".join(tokens)
    ascii_view = normalized.lower()

    for pattern, replacement in SYNONYM_PATTERNS:
        ascii_view = pattern.sub(replacement, ascii_view)

    # ascii_view may strip casing but Indian scripts remain intact after lower().
    return ascii_view


def _count_chars_in_range(text: str, unicode_range: tuple[int, int]) -> int:
    """Count characters that fall within a Unicode block range."""

    start, end = unicode_range
    return sum(1 for ch in text if start <= ord(ch) <= end)


def combined_language(lang: Language | None, detected: Language) -> Language:
    """Resolve UI override language and detected language."""

    if lang and lang != Language.AUTO:
        return lang
    return detected


def available_languages() -> list[Language]:
    """List of selectable languages."""

    return [Language.AUTO, Language.EN, Language.HI, Language.TA]
