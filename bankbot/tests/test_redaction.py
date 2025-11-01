"""Tests for PII redaction utilities."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.api import app, audit_logger, session_memory
from app.core.redaction import redact_text, restore_tokens


@pytest.mark.parametrize(
    "text,pattern",
    [
        ("Call me at 9876543210", "9876543210"),
        ("Email me test.user@example.com", "test.user@example.com"),
        ("PAN is ABCDE1234F", "ABCDE1234F"),
        ("My account number is 123456789012", "123456789012"),
        ("Card 4111 1111 1111 1111", "4111 1111 1111 1111"),
        ("Aadhaar 1234 5678 9123", "1234 5678 9123"),
    ],
)
def test_redaction_replaces_pii(text: str, pattern: str) -> None:
    result = redact_text(text)
    assert pattern not in result.sanitized_text
    assert result.token_map
    round_trip = restore_tokens(result.sanitized_text, result.token_map)
    assert pattern.replace(" ", "") in round_trip.replace(" ", "")


def test_audit_log_avoids_raw_pii(tmp_path: Path) -> None:
    # Redirect audit logs to a temp file to isolate test
    log_path = tmp_path / "audit.log"
    audit_logger.path = log_path
    session_memory.tokens.clear()
    session_memory.turns.clear()

    client = TestClient(app)
    client.post(
        "/chat/turn",
        json={
            "message": "Please block card 4111 1111 1111 1111",
            "lang": "AUTO",
            "confidence_threshold": 0.18,
            "use_sentence_transformers": False,
        },
    )

    log_contents = log_path.read_text(encoding="utf-8")
    assert "4111" not in log_contents
    assert "__CARD_" in log_contents
