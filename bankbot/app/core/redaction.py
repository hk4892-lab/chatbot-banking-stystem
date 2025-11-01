"""PII redaction utilities with deterministic reversible tokens."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Dict, Iterable


PHONE_PATTERN = re.compile(r"(\+91[- ]?)?[6-9]\d{9}")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PAN_PATTERN = re.compile(r"[A-Z]{5}\d{4}[A-Z]", re.IGNORECASE)
AADHAAR_PATTERN = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}\b")

# Card numbers: 12-19 digits allowing separators
CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){12,19}\d(?!\d)")
# Account numbers: 9-18 digits without separators to avoid double matching
ACCOUNT_PATTERN = re.compile(r"(?<!\d)\d{9,18}(?!\d)")


PATTERN_ORDER: Iterable[tuple[str, re.Pattern[str]]] = (
    ("CARD", CARD_PATTERN),
    ("ACCOUNT", ACCOUNT_PATTERN),
    ("PHONE", PHONE_PATTERN),
    ("EMAIL", EMAIL_PATTERN),
    ("PAN", PAN_PATTERN),
    ("AADHAAR", AADHAAR_PATTERN),
)


@dataclass
class RedactionResult:
    """Container for redacted text and reversible token map."""

    sanitized_text: str
    token_map: Dict[str, str]


def _token_for(label: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"__{label}_{digest}__"


def redact_text(text: str) -> RedactionResult:
    """Redact PII values in text and return mapping of reversible tokens."""

    token_map: Dict[str, str] = {}

    def replace(match: re.Match[str], label: str) -> str:
        value = match.group(0)
        token = _token_for(label, value)
        token_map.setdefault(token, value)
        return token

    sanitized = text
    for label, pattern in PATTERN_ORDER:
        sanitized = pattern.sub(lambda m, lbl=label: replace(m, lbl), sanitized)

    return RedactionResult(sanitized_text=sanitized, token_map=token_map)


def restore_tokens(value: str, token_map: Dict[str, str]) -> str:
    """Restore tokens in the provided value using the reversible map."""

    for token, original in token_map.items():
        value = value.replace(token, original)
    return value


def serialize_token_map(token_map: Dict[str, str]) -> str:
    """Serialize token map for logging without exposing raw values."""

    # Only persist token keys; actual values are not logged.
    return json.dumps(sorted(token_map.keys()))
