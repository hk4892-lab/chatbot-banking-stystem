"""Dependency helpers for FastAPI wiring."""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .config import Settings, get_settings
from .generator import GroundedGenerator
from .pipeline import ChatPipeline
from .retrieval import FaissRetriever


def _resolve_device(settings: Settings) -> str:
    if settings.device_preference == 'cpu':
        return 'cpu'
    if settings.device_preference == 'cuda':
        try:
            import torch  # noqa: WPS433

            if torch.cuda.is_available():
                return 'cuda'
        except Exception:  # pragma: no cover
            return 'cpu'
    try:
        import torch

        if torch.cuda.is_available():
            return 'cuda'
    except Exception:  # pragma: no cover
        return 'cpu'
    return 'cpu'


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    settings = get_settings()
    device = _resolve_device(settings)
    return SentenceTransformer(settings.embedding_model, device=device)


@lru_cache(maxsize=1)
def get_retriever() -> FaissRetriever:
    settings = get_settings()
    embedding_model = get_embedding_model()
    return FaissRetriever(settings=settings, embedding_model=embedding_model)


@lru_cache(maxsize=1)
def get_generator() -> GroundedGenerator:
    settings = get_settings()
    return GroundedGenerator(settings=settings)


@lru_cache(maxsize=1)
def get_audit_logger() -> logging.Logger:
    settings = get_settings()
    logger = logging.getLogger('audit')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(settings.audit_log_path, encoding='utf-8')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


@lru_cache(maxsize=1)
def get_pipeline() -> ChatPipeline:
    settings = get_settings()
    retriever = get_retriever()
    generator = get_generator()
    audit_logger = get_audit_logger()
    return ChatPipeline(settings=settings, retriever=retriever, generator=generator, audit_logger=audit_logger)
