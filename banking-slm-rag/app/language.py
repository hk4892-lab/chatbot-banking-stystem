"""Utilities for language detection and normalization."""

from __future__ import annotations

import re
from typing import Dict


LANG_EN = 'EN'
LANG_HI = 'HI'
LANG_TA = 'TA'

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
TAMIL_RE = re.compile(r'[\u0B80-\u0BFF]')

_BASE_SYNONYMS: Dict[str, Dict[str, str]] = {
    LANG_EN: {
        'otp': 'one time password',
        'stmt': 'statement',
        'acc': 'account',
    },
    LANG_HI: {
        'kitna': 'kya limit',
        'balance batao': 'balance janana hai',
        'paisa': 'amount',
        'statement download': 'statement download',
    },
    LANG_TA: {
        'pannunga': 'dayavuseithu',
        'engaluku': 'namakku',
        'balance sollunga': 'balance terinchikanum',
        'pin reset': 'pin maru seyyavum',
    },
}

_CODE_MIX_SYNONYMS: Dict[str, Dict[str, str]] = {
    LANG_EN: {
        'how much limit': 'limit details',
        'debit card pin': 'debit card pin reset',
    },
    LANG_HI: {
        'upi limit': 'upi seema',
        'statement download karo': 'statement download',
        'kitna limit': 'limit kitna',
    },
    LANG_TA: {
        'upi limit': 'upi varambu',
        'hotlist pannunga': 'card block seyya',
        'statement download pannalaam': 'statement eduka',
    },
}


def detect_lang(text: str) -> str:
    """Detect language using Unicode script heuristics."""

    if not text:
        return LANG_EN
    has_devanagari = bool(DEVANAGARI_RE.search(text))
    has_tamil = bool(TAMIL_RE.search(text))
    if has_tamil and not has_devanagari:
        return LANG_TA
    if has_devanagari and not has_tamil:
        return LANG_HI
    # Mixed scripts: prefer Tamil if more matches, else Hindi.
    devanagari_count = len(DEVANAGARI_RE.findall(text))
    tamil_count = len(TAMIL_RE.findall(text))
    if tamil_count > devanagari_count:
        return LANG_TA
    if devanagari_count > 0:
        return LANG_HI
    return LANG_EN


def normalize(text: str, lang: str) -> str:
    """Normalize text for retrieval purposes."""

    lowered = text.lower().strip()
    collapsed = re.sub(r'\s+', ' ', lowered)
    synonyms = {**_BASE_SYNONYMS.get(lang, {}), **_CODE_MIX_SYNONYMS.get(lang, {})}
    normalized = collapsed
    for key, value in synonyms.items():
        normalized = normalized.replace(key, value)
    return normalized


def available_langs() -> Dict[str, str]:
    """Return available language display names."""

    return {
        LANG_EN: 'English',
        LANG_HI: 'Hindi',
        LANG_TA: 'Tamil',
    }
