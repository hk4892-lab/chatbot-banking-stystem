"""PII redaction utilities for BankBot."""

from __future__ import annotations

import re
from typing import Iterable


_PATTERNS: Iterable[tuple[str, re.Pattern[str]]] = (
    ("CARD", re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")),
    ("ACCOUNT", re.compile(r"\b\d{10,12}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"\b(?:\+91[- ]?)?[6-9]\d{9}\b")),
)


def redact(text: str | None) -> str:
    """Mask known PII tokens with redaction placeholders."""

    if not text:
        return ""

    redacted = text
    for label, pattern in _PATTERNS:
        redacted = pattern.sub(f"[REDACTED:{label}]", redacted)
    return redacted


def safe_for_log(text: str | None) -> str:
    """Return a redacted string safe for logging."""

    return redact(text)


__all__ = ["redact", "safe_for_log"]
