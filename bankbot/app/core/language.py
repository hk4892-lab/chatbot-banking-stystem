"""Language detection and normalization utilities."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List

# Unicode script ranges
DEVANAGARI_RANGE = (0x0900, 0x097F)
TAMIL_RANGE = (0x0B80, 0x0BFF)


SYNONYM_MAP: Dict[str, List[str]] = {
    "limit kitna": ["upi limit"],
    "upi kitna": ["upi limit"],
    "statement bhej": ["statement request"],
    "statement send pannunga": ["statement request"],
    "emi calc": ["emi calculation"],
    "emi calculate": ["emi calculation"],
    "emi calc karna": ["emi calculation"],
    "card block": ["block card", "debit card blocking"],
    "block card": ["block card", "debit card blocking"],
    "block my card": ["block card", "debit card blocking"],
    "card limit": ["credit card limit"],
    "kyc kab": ["kyc update"],
    "kyc update": ["kyc update"],
    "statement send": ["statement request"],
    "statement copy": ["statement request"],
    "statement please": ["statement request"],
    "emi kitna": ["emi calculation"],
    "emi amount": ["emi calculation"],
    "emi calc pannunga": ["emi calculation"],
    "limit vandhu": ["credit card limit"],
    "limit venum": ["upi limit"],
}


@dataclass
class LanguageDetection:
    """Result of script-based language detection."""

    primary: str
    detected_scripts: List[str]
    is_mixed: bool


def _has_char_in_range(text: str, start: int, end: int) -> bool:
    return any(start <= ord(ch) <= end for ch in text)


def detect_language(text: str) -> LanguageDetection:
    """Detect the primary language for the given text based on script ranges."""

    has_devanagari = _has_char_in_range(text, *DEVANAGARI_RANGE)
    has_tamil = _has_char_in_range(text, *TAMIL_RANGE)
    scripts: List[str] = []
    primary = "EN"

    if has_devanagari:
        scripts.append("HI")
    if has_tamil:
        scripts.append("TA")
    if not scripts:
        scripts.append("EN")

    if has_devanagari and not has_tamil:
        primary = "HI"
    elif has_tamil and not has_devanagari:
        primary = "TA"
    elif has_devanagari and has_tamil:
        primary = "EN"  # fallback for mixed Indian scripts

    return LanguageDetection(primary=primary, detected_scripts=scripts, is_mixed=len(set(scripts)) > 1)


LATIN_REGEX = re.compile(r"[A-Za-z]")


def normalize_text(text: str) -> str:
    """Normalize the query for retrieval handling code-mixed content."""

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    normalized_tokens: List[str] = []
    for token in text.split():
        if LATIN_REGEX.search(token):
            normalized_tokens.append(token.lower())
        else:
            normalized_tokens.append(token)

    normalized = " ".join(normalized_tokens)

    lower_normalized = normalized.lower()
    expansions: List[str] = []
    for trigger, synonyms in SYNONYM_MAP.items():
        if trigger in lower_normalized:
            expansions.extend(synonyms)

    if expansions:
        normalized = normalized + " " + " ".join(sorted(set(expansions)))

    return normalized
