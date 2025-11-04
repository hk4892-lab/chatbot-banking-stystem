"""Tests for PII redaction."""
import pytest
from app.redaction import redact, scrub_log, has_sensitive_keywords


def test_redact_phone():
    """Test phone number redaction."""
    text = "Call me at 9876543210"
    masked, mapping = redact(text)
    
    assert "9876543210" not in masked
    assert "PII_PHONE_1" in masked
    assert mapping["PII_PHONE_1"] == "9876543210"


def test_redact_email():
    """Test email redaction."""
    text = "Email me at user@example.com for help"
    masked, mapping = redact(text)
    
    assert "user@example.com" not in masked
    assert "PII_EMAIL_1" in masked
    assert mapping["PII_EMAIL_1"] == "user@example.com"


def test_redact_pan():
    """Test PAN number redaction."""
    text = "My PAN is ABCDE1234F"
    masked, mapping = redact(text)
    
    assert "ABCDE1234F" not in masked
    assert "PII_PAN_1" in masked
    assert mapping["PII_PAN_1"] == "ABCDE1234F"


def test_redact_aadhaar():
    """Test Aadhaar number redaction."""
    text = "Aadhaar: 1234 5678 9012"
    masked, mapping = redact(text)
    
    assert "1234 5678 9012" not in masked
    assert "PII_AADHAAR_1" in masked


def test_redact_card():
    """Test card number redaction."""
    text = "Card number 1234-5678-9012-3456"
    masked, mapping = redact(text)
    
    assert "1234-5678-9012-3456" not in masked
    assert "PII_CARD_1" in masked


def test_redact_account():
    """Test account number redaction."""
    text = "Transfer to account 123456789012"
    masked, mapping = redact(text)
    
    assert "123456789012" not in masked
    assert "PII_ACCOUNT_1" in masked


def test_redact_multiple():
    """Test multiple PII types in one text."""
    text = "Call 9876543210 or email user@example.com with account 123456789012"
    masked, mapping = redact(text)
    
    assert "9876543210" not in masked
    assert "user@example.com" not in masked
    assert "123456789012" not in masked
    assert len(mapping) == 3


def test_scrub_log():
    """Test log scrubbing removes PII."""
    record = {
        "query": "My phone is 9876543210",
        "email": "test@example.com",
        "nested": {"phone": "9988776655"},
    }
    
    scrubbed = scrub_log(record)
    
    assert "9876543210" not in scrubbed["query"]
    assert "test@example.com" not in scrubbed["email"]
    assert "9988776655" not in scrubbed["nested"]["phone"]
    assert "PII_PHONE" in scrubbed["query"]
    assert "PII_EMAIL" in scrubbed["email"]


def test_has_sensitive_keywords():
    """Test detection of forbidden banking keywords."""
    assert has_sensitive_keywords("show my balance")
    assert has_sensitive_keywords("transfer money")
    assert has_sensitive_keywords("send OTP")
    assert has_sensitive_keywords("मेरा बैलेंस बताओ")
    assert has_sensitive_keywords("பணம் அனுப்பு")
    assert not has_sensitive_keywords("how to reset PIN")
    assert not has_sensitive_keywords("UPI transaction limit")


def test_no_pii():
    """Test text without PII passes through."""
    text = "What is the UPI transaction limit?"
    masked, mapping = redact(text)
    
    assert masked == text
    assert len(mapping) == 0
