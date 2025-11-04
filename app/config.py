"""Configuration management using pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Model Configuration
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    slm_model: str = "microsoft/phi-3-mini-4k-instruct"

    # Index Paths
    faiss_index_path: str = "index/kb.faiss"
    faiss_meta_path: str = "index/kb.meta.pkl"

    # KB Data
    kb_chunks_path: str = "data/kb_chunks.sample.jsonl"

    # Generation Defaults
    default_k: int = 5
    default_temperature: float = 0.3
    default_top_p: float = 0.9
    default_max_tokens: int = 256

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost,http://localhost:8501"

    # Safety
    refusal_threshold: float = 0.2
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    # Logging
    log_level: str = "INFO"
    audit_log_path: str = "logs/audit.jsonl"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins as list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


# Global settings instance
settings = Settings()
