"""Configuration management for BankBot."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _as_bool(value: str | None, *, default: bool = True) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


USE_SLM: bool = _as_bool(os.getenv("USE_SLM"), default=False)
MODEL_ID: str = os.getenv("MODEL_ID", "microsoft/phi-3-mini-4k-instruct")
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8000"))

LOG_PATH: Path = Path(os.getenv("LOG_PATH", "bankbot/logs/audit.log"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


__all__ = [
    "USE_SLM",
    "MODEL_ID",
    "HOST",
    "PORT",
    "LOG_PATH",
]
