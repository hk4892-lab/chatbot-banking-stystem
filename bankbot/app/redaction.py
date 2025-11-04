"""PII redaction utilities for safe logging."""

from __future__ import annotations

import re
from typing import Iterable


_REPLACEMENTS: Iterable[tuple[str, re.Pattern[str]]] = (
    ("CARD", re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")),
    ("ACCOUNT", re.compile(r"\b\d{10,12}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    (
        "PHONE",
        re.compile(r"\b(?:\+91[- ]?)?[6-9]\d{9}\b"),
    ),
)


def redact(text: str | None) -> str:
    """Replace PII patterns with redaction tokens."""

    if not text:
        return ""

    redacted = text
    for label, pattern in _REPLACEMENTS:
        redacted = pattern.sub(f"[REDACTED:{label}]", redacted)
    return redacted


def redact_for_logs(text: str | None) -> str:
    """Alias for redact to clarify intent when logging."""

    return redact(text)


__all__ = ["redact", "redact_for_logs"]
