"""Language detection, normalization, and code-mix handling."""
import re
from typing import Literal


# Unicode ranges for script detection
DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
TAMIL_RANGE = re.compile(r"[\u0B80-\u0BFF]")


# Code-mix synonyms for Hinglish and Tamlish
HINGLISH_SYNONYMS = {
    "kitna": "how much",
    "kaise": "how",
    "kya": "what",
    "kyun": "why",
    "kab": "when",
    "kahan": "where",
    "bank": "bank",
    "account": "account",
    "card": "card",
    "paisa": "money",
    "paise": "money",
    "balance": "balance",
    "transfer": "transfer",
}

TAMLISH_SYNONYMS = {
    "eppadi": "how",
    "enna": "what",
    "enge": "where",
    "eppo": "when",
    "pannunga": "please",
    "sollunga": "tell",
    "account": "account",
    "card": "card",
    "bank": "bank",
    "panam": "money",
}


def detect_lang(text: str) -> Literal["EN", "HI", "TA"]:
    """
    Detect language based on Unicode script ranges.
    
    Args:
        text: Input text
        
    Returns:
        Language code: EN, HI, or TA
    """
    # Count characters in each script
    devanagari_count = len(DEVANAGARI_RANGE.findall(text))
    tamil_count = len(TAMIL_RANGE.findall(text))
    
    # If significant non-Latin script present, classify accordingly
    if devanagari_count > 3:
        return "HI"
    elif tamil_count > 3:
        return "TA"
    else:
        return "EN"


def normalize(text: str, lang: Literal["EN", "HI", "TA"]) -> str:
    """
    Normalize text with language-specific processing.
    
    Args:
        text: Input text
        lang: Detected language
        
    Returns:
        Normalized text
    """
    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", text).strip()
    
    # Apply synonyms based on language
    text_lower = normalized.lower()
    
    if lang == "EN" or lang == "HI":
        # Apply Hinglish synonyms
        for hinglish, english in HINGLISH_SYNONYMS.items():
            if hinglish in text_lower:
                # Replace while preserving case context
                normalized = re.sub(
                    r"\b" + hinglish + r"\b",
                    english,
                    normalized,
                    flags=re.IGNORECASE
                )
    
    if lang == "EN" or lang == "TA":
        # Apply Tamlish synonyms
        for tamlish, english in TAMLISH_SYNONYMS.items():
            if tamlish in text_lower:
                normalized = re.sub(
                    r"\b" + tamlish + r"\b",
                    english,
                    normalized,
                    flags=re.IGNORECASE
                )
    
    return normalized


def get_refusal_template(lang: Literal["EN", "HI", "TA"]) -> str:
    """
    Get localized refusal message.
    
    Args:
        lang: Language code
        
    Returns:
        Refusal message in specified language
    """
    templates = {
        "EN": (
            "I apologize, but I don't have enough information to answer your question accurately. "
            "Please contact our customer service at 1800-XXX-XXXX or visit your nearest branch "
            "for assistance with account-specific queries."
        ),
        "HI": (
            "क्षमा करें, मेरे पास आपके प्रश्न का सटीक उत्तर देने के लिए पर्याप्त जानकारी नहीं है। "
            "कृपया 1800-XXX-XXXX पर हमारी ग्राहक सेवा से संपर्क करें या अपनी निकटतम शाखा में जाएं।"
        ),
        "TA": (
            "மன்னிக்கவும், உங்கள் கேள்விக்கு துல்லியமாக பதிலளிக்க என்னிடம் போதுமான தகவல் இல்லை. "
            "தயவுசெய்து 1800-XXX-XXXX என்ற எண்ணில் எங்கள் வாடிக்கையாளர் சேவையைத் தொடர்பு கொள்ளவும்."
        ),
    }
    return templates.get(lang, templates["EN"])
