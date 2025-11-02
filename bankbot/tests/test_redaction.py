from __future__ import annotations

from pathlib import Path

from app.core.redaction import PIIRedactor, tokens_only


def test_redaction_replaces_and_restores() -> None:
    text = "Call me at +919876543210 or email demo.user@example.com"
    redactor = PIIRedactor()

    clean, mapping = redactor.redact(text)

    assert "+919876543210" not in clean
    assert "demo.user@example.com" not in clean
    assert mapping

    for token, original in mapping.items():
        restored = PIIRedactor.detokenize(token, mapping)
        assert restored == original
        assert token in clean


def test_audit_log_contains_only_tokens(monkeypatch, tmp_path: Path) -> None:
    from app import api

    redactor = PIIRedactor()
    _, mapping = redactor.redact("Block card 4111 1111 1111 1111")
    token = next(iter(mapping))

    fake_log = tmp_path / "audit.log"
    monkeypatch.setattr(api, "LOG_PATH", fake_log)

    api._write_audit_log({
        "ts": "2024-01-01T00:00:00Z",
        "session": "test",
        "lang": "EN",
        "route": "tool",
        "topdoc_id": None,
        "top_score": 0.0,
        "confidence": 0.0,
        "tools_called": ["block_card"],
        "redaction_tokens": tokens_only(mapping),
        "ip": None,
    })

    contents = fake_log.read_text(encoding="utf-8")
    assert "4111" not in contents
    assert token in contents
