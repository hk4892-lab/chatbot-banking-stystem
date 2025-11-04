"""Deterministic PII masking utilities."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Tuple


_PATTERNS = {
    'PHONE': re.compile(r'(?:\+91[- ]?)?[6-9]\d{9}'),
    'EMAIL': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
    'PAN': re.compile(r'([A-Z]{5}[0-9]{4}[A-Z]{1})', re.IGNORECASE),
    'AADHAAR': re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}\b'),
    'CARD': re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{0,4}\b'),
}


def _make_token(kind: str, counter: int) -> str:
    return f'PII_{kind}_{counter}'


def redact(text: str) -> Tuple[str, Dict[str, str]]:
    """Mask sensitive entities with deterministic tokens."""

    mapping: Dict[str, str] = {}
    counters: Dict[str, int] = {key: 0 for key in _PATTERNS.keys()}
    redacted = text

    for kind, pattern in _PATTERNS.items():
        def _repl(match: re.Match[str]) -> str:
            counters[kind] += 1
            token = _make_token(kind, counters[kind])
            mapping[token] = match.group(0)
            return token

        redacted = pattern.sub(_repl, redacted)

    return redacted, mapping


def _scrub_value(value: Any) -> Any:
    if isinstance(value, str):
        masked, _ = redact(value)
        return masked
    if isinstance(value, dict):
        return {key: _scrub_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_scrub_value(v) for v in value]
    return value


def scrub_log(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a sanitized copy of a log record without raw PII."""

    return {key: _scrub_value(value) for key, value in record.items()}


def iter_sensitive_tokens(text: str) -> Iterable[str]:
    """Yield masked tokens present in the text."""

    return (token for token in text.split() if token.startswith('PII_'))
