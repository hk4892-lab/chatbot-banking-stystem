"""Centralized logging utilities with enforced redaction."""

from __future__ import annotations

import logging
from typing import Any, Dict

from .config import LOG_PATH
from .redaction import redact_for_logs


_LOGGER = logging.getLogger("bankbot")
_LOGGER.setLevel(logging.INFO)

if not _LOGGER.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    _LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    _LOGGER.addHandler(stream_handler)


def _sanitize_context(context: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in context.items():
        sanitized[key] = redact_for_logs(str(value))
    return sanitized


def log_info(event: str, message: str | None = None, **context: Any) -> None:
    """Log an info-level event with sanitized message and context."""

    sanitized_message = redact_for_logs(message or "")
    sanitized_context = _sanitize_context(context)

    if sanitized_context:
        _LOGGER.info("event=%s message=%s context=%s", event, sanitized_message, sanitized_context)
    else:
        _LOGGER.info("event=%s message=%s", event, sanitized_message)


def get_logger() -> logging.Logger:
    return _LOGGER


__all__ = ["log_info", "get_logger"]
