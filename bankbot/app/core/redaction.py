"""Deterministic PII redaction with reversible tokens."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable


TokenMap = Dict[str, str]


@dataclass(frozen=True)
class PiiPattern:
    """PII pattern definition with a type label."""

    name: str
    regex: re.Pattern[str]


PII_PATTERNS: tuple[PiiPattern, ...] = (
    PiiPattern(
        "CARD",
        re.compile(r"\b(?:\d[ -]?){12,19}\b"),
    ),
    PiiPattern(
        "AADHAAR",
        re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    ),
    PiiPattern(
        "ACCOUNT",
        re.compile(r"\b\d{9,18}\b"),
    ),
    PiiPattern(
        "PHONE",
        re.compile(r"\b(?:\+91[- ]?)?[6-9]\d{9}\b"),
    ),
    PiiPattern(
        "EMAIL",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    ),
    PiiPattern(
        "PAN",
        re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    ),
)


class PIIRedactor:
    """Redact PII by replacing known patterns with reversible tokens."""

    def __init__(self, patterns: Iterable[PiiPattern] | None = None) -> None:
        self._patterns = tuple(patterns) if patterns else PII_PATTERNS

    def redact(self, text: str) -> tuple[str, TokenMap]:
        """Redact the given text and return the mapping for later restoration."""

        token_map: TokenMap = {}
        redacted = text

        for pattern in self._patterns:
            redacted = pattern.regex.sub(
                lambda match, *, pattern=pattern: self._token_for(
                    pattern.name, match.group(0), token_map
                ),
                redacted,
            )

        return redacted, token_map

    @staticmethod
    def _token_for(kind: str, value: str, mapping: TokenMap) -> str:
        digest = hashlib.sha256(value.encode()).hexdigest()[:12].upper()
        token = f"[[{kind}_{digest}]]"
        mapping.setdefault(token, value)
        return token

    @staticmethod
    def detokenize(token: str, mapping: TokenMap) -> str:
        """Restore the original value for a specific token."""

        if token not in mapping:
            raise KeyError(f"Token {token} not present in map")
        return mapping[token]

    @staticmethod
    def detokenize_text(text: str, mapping: TokenMap) -> str:
        """Replace tokens present in the text using the provided token map."""

        if not mapping:
            return text

        token_regex = re.compile(
            "|".join(re.escape(token) for token in sorted(mapping.keys(), key=len, reverse=True))
        )

        return token_regex.sub(lambda match: mapping.get(match.group(0), match.group(0)), text)


def tokens_only(mapping: TokenMap) -> list[str]:
    """Helper to list token keys without exposing values."""

    return sorted(mapping.keys())
