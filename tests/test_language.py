"""Tests for language detection and normalization."""
import pytest
from app.language import detect_lang, normalize, get_refusal_template


def test_detect_lang_english():
    """Test English detection."""
    assert detect_lang("How to reset PIN?") == "EN"
    assert detect_lang("What is the UPI limit?") == "EN"


def test_detect_lang_hindi():
    """Test Hindi detection."""
    assert detect_lang("एटीएम पिन कैसे रीसेट करें?") == "HI"
    assert detect_lang("यूपीआई की सीमा क्या है?") == "HI"


def test_detect_lang_tamil():
    """Test Tamil detection."""
    assert detect_lang("எப்படி கார்டை பிளாக் செய்வது?") == "TA"
    assert detect_lang("என் கணக்கு எண் என்ன?") == "TA"


def test_detect_lang_hinglish():
    """Test Hinglish defaults to English."""
    assert detect_lang("UPI ki limit kitni hai?") == "EN"


def test_normalize_whitespace():
    """Test whitespace normalization."""
    text = "How   to    reset   PIN?"
    assert normalize(text, "EN") == "How to reset PIN?"


def test_normalize_hinglish_synonyms():
    """Test Hinglish synonym replacement."""
    text = "UPI ki limit kitna hai?"
    normalized = normalize(text, "EN")
    assert "how much" in normalized.lower()


def test_normalize_tamlish_synonyms():
    """Test Tamlish synonym replacement."""
    text = "Please tell me eppadi to do this"
    normalized = normalize(text, "TA")
    assert "how" in normalized.lower()


def test_get_refusal_template_english():
    """Test English refusal template."""
    template = get_refusal_template("EN")
    assert "apologize" in template.lower()
    assert "1800" in template


def test_get_refusal_template_hindi():
    """Test Hindi refusal template."""
    template = get_refusal_template("HI")
    assert "क्षमा" in template
    assert "1800" in template


def test_get_refusal_template_tamil():
    """Test Tamil refusal template."""
    template = get_refusal_template("TA")
    assert "மன்னிக்கவும்" in template
    assert "1800" in template
