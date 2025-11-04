"""Application configuration using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the chatbot service."""

    model_config = SettingsConfigDict(env_file=('.env',), env_file_encoding='utf-8', case_sensitive=False)

    embedding_model: str = Field(
        default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        description='SentenceTransformers model used for encoding queries and passages.',
    )
    slm_model: str = Field(
        default='microsoft/phi-3-mini-4k-instruct',
        description='Small language model used for grounded generation.',
    )
    faiss_index_path: Path = Field(
        default=Path('index/kb.faiss'),
        description='Filesystem path to the FAISS index file.',
    )
    faiss_meta_path: Path = Field(
        default=Path('index/kb.meta.pkl'),
        description='Filesystem path to the FAISS metadata pickle.',
    )
    kb_source_path: Path = Field(
        default=Path('data/kb_chunks.sample.jsonl'),
        description='Fallback knowledge base JSONL used if FAISS files missing.',
    )
    default_k: int = Field(default=5, ge=1, le=20)
    default_temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    min_relevance: float = Field(
        default=0.2,
        description='Minimum FAISS cosine score required to treat a passage as relevant.',
    )
    refusal_threshold: float = Field(
        default=0.18,
        description='If top score falls below this threshold, the system refuses.',
    )
    rate_limit_rps: float = Field(default=2.0, ge=0.1)
    rate_limit_burst: int = Field(default=4, ge=1)
    app_env: Literal['development', 'staging', 'production'] = Field(default='development')
    audit_log_path: Path = Field(default=Path('logs/audit.log'))
    max_prompt_context_tokens: int = Field(default=2048)
    device_preference: Literal['auto', 'cpu', 'cuda'] = Field(default='auto')


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance."""

    settings = Settings()
    audit_dir = settings.audit_log_path.parent
    if audit_dir and not audit_dir.exists():
        audit_dir.mkdir(parents=True, exist_ok=True)
    return settings
