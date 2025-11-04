from app.redaction import redact, scrub_log


def test_redact_masks_sensitive_patterns() -> None:
    text = 'Call me at 9123456789 or email user@example.com. Card 1234 5678 9012 3456 PAN ABCDE1234F.'
    masked, mapping = redact(text)
    assert '9123456789' not in masked
    assert 'user@example.com' not in masked
    assert '1234 5678 9012 3456' not in masked
    assert 'ABCDE1234F' not in masked
    assert any(token.startswith('PII_PHONE') for token in mapping)
    assert any(token.startswith('PII_EMAIL') for token in mapping)


def test_scrub_log_removes_pii() -> None:
    record = {'message': 'OTP sent to 9876543210', 'metadata': {'email': 'audit@example.com'}}
    scrubbed = scrub_log(record)
    assert '9876543210' not in scrubbed['message']
    assert 'audit@example.com' not in scrubbed['metadata']['email']
