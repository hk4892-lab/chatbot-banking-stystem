"""Configuration helpers for BankBot."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


USE_SLM: bool = _as_bool(os.getenv("USE_SLM"), default=True)
MODEL_ID: str = os.getenv("MODEL_ID", "microsoft/phi-3-mini-4k-instruct")
EMBED_MODEL: str = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = _as_int(os.getenv("PORT"), 8000)

LOG_PATH: Path = Path(os.getenv("LOG_PATH", "bankbot/logs/audit.log"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

TOP_K: int = _as_int(os.getenv("TOP_K"), 4)
RAG_SCORE_THRESHOLD: float = _as_float(os.getenv("RAG_SCORE_THRESHOLD"), 0.32)


__all__ = [
    "USE_SLM",
    "MODEL_ID",
    "EMBED_MODEL",
    "HOST",
    "PORT",
    "LOG_PATH",
    "TOP_K",
    "RAG_SCORE_THRESHOLD",
]
