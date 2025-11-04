"""Pydantic schemas for API requests and responses."""
from typing import Literal
from pydantic import BaseModel, Field


class ChatTurnRequest(BaseModel):
    """Request schema for /chat/turn endpoint."""

    text: str = Field(..., min_length=1, max_length=2000, description="User query text")
    k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    lang: Literal["AUTO", "EN", "HI", "TA"] = Field(
        default="AUTO", description="Language hint (AUTO for detection)"
    )
    temperature: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Generation temperature"
    )


class Citation(BaseModel):
    """Citation reference."""

    doc_id: str
    title: str


class SafetyInfo(BaseModel):
    """Safety metadata."""

    pii_masked: bool
    refusal: bool


class ChatTurnResponse(BaseModel):
    """Response schema for /chat/turn endpoint."""

    reply: str
    citations: list[Citation]
    used_context_ids: list[str]
    route: Literal["answer", "refusal"]
    latency_ms: int
    lang: Literal["EN", "HI", "TA"]
    safety: SafetyInfo


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"


class ConfigResponse(BaseModel):
    """Configuration info response."""

    embedding_model: str
    slm_model: str
    default_k: int
    default_temperature: float
    languages: list[str]
    refusal_threshold: float
