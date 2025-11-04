"""Dependency injection and singleton management for FastAPI."""
import json
import os
from pathlib import Path
from typing import Optional

from app.config import settings
from app.retrieval import FAISSRetriever
from app.generator import SLMGenerator
from app.pipeline import RAGPipeline


# Global singletons (lazy-loaded)
_retriever: Optional[FAISSRetriever] = None
_generator: Optional[SLMGenerator] = None
_pipeline: Optional[RAGPipeline] = None


def get_retriever() -> FAISSRetriever:
    """Get or create FAISSRetriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = FAISSRetriever(
            index_path=settings.faiss_index_path,
            meta_path=settings.faiss_meta_path,
            embedding_model=settings.embedding_model,
        )
    return _retriever


def get_generator() -> SLMGenerator:
    """Get or create SLMGenerator singleton."""
    global _generator
    if _generator is None:
        _generator = SLMGenerator(model_name=settings.slm_model)
    return _generator


def get_pipeline() -> RAGPipeline:
    """Get or create RAGPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        retriever = get_retriever()
        generator = get_generator()
        _pipeline = RAGPipeline(retriever=retriever, generator=generator)
    return _pipeline


def ensure_audit_log_dir():
    """Ensure audit log directory exists."""
    log_path = Path(settings.audit_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)


def write_audit_log(record: dict):
    """Write audit log entry."""
    ensure_audit_log_dir()
    with open(settings.audit_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
