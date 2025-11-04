"""Deterministic PII redaction for banking security."""
import re
from typing import Dict, Tuple


# Regex patterns for various PII types
PATTERNS = {
    "phone": re.compile(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "aadhaar": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "account": re.compile(r"\b\d{12,16}\b"),
}


def redact(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Redact PII from text with deterministic tokens.
    
    Args:
        text: Input text potentially containing PII
        
    Returns:
        Tuple of (masked_text, mapping_dict)
    """
    masked_text = text
    mapping = {}
    
    # Track counters for each PII type
    counters = {key: 1 for key in PATTERNS.keys()}
    
    # Process each pattern type
    for pii_type, pattern in PATTERNS.items():
        matches = pattern.finditer(masked_text)
        for match in matches:
            original = match.group()
            token = f"PII_{pii_type.upper()}_{counters[pii_type]}"
            mapping[token] = original
            masked_text = masked_text.replace(original, token, 1)
            counters[pii_type] += 1
    
    return masked_text, mapping


def scrub_log(record: dict) -> dict:
    """
    Remove raw PII from log records.
    
    Args:
        record: Log record dictionary
        
    Returns:
        Scrubbed log record
    """
    scrubbed = record.copy()
    
    # Recursively scrub string values
    for key, value in scrubbed.items():
        if isinstance(value, str):
            scrubbed[key], _ = redact(value)
        elif isinstance(value, dict):
            scrubbed[key] = scrub_log(value)
        elif isinstance(value, list):
            scrubbed[key] = [
                scrub_log(item) if isinstance(item, dict) else redact(item)[0] if isinstance(item, str) else item
                for item in value
            ]
    
    return scrubbed


def has_sensitive_keywords(text: str) -> bool:
    """
    Check if text contains forbidden banking action keywords.
    
    Args:
        text: Input text
        
    Returns:
        True if sensitive keywords found
    """
    text_lower = text.lower()
    forbidden_keywords = [
        "balance", "transaction", "transfer", "otp", "pin", "password",
        "send money", "pay", "withdraw", "deposit",
        # Hindi equivalents
        "बैलेंस", "लेन-देन", "ट्रांसफर", "पैसे भेजो",
        # Tamil equivalents
        "பணம்", "பரிமாற்றம்", "அனுப்பு",
    ]
    
    return any(keyword in text_lower for keyword in forbidden_keywords)
