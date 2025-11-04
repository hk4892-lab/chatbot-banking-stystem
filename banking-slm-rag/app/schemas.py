"""Pydantic schema definitions for the API."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatTurnRequest(BaseModel):
    text: str = Field(..., min_length=1, description='Latest user utterance.')
    k: int = Field(5, ge=1, le=10, description='Number of passages to retrieve.')
    lang: Literal['AUTO', 'EN', 'HI', 'TA'] = Field('AUTO', description='Requested answer language.')
    temperature: float = Field(0.3, ge=0.0, le=1.0, description='Sampling temperature for the SLM.')


class Citation(BaseModel):
    doc_id: str
    title: str


class SafetyFlags(BaseModel):
    pii_masked: bool
    refusal: bool


class ChatTurnResponse(BaseModel):
    reply: str
    citations: List[Citation]
    used_context_ids: List[str]
    route: Literal['answer', 'refusal']
    latency_ms: int
    lang: Literal['EN', 'HI', 'TA']
    safety: SafetyFlags


class ConfigResponse(BaseModel):
    embedding_model: str
    slm_model: str
    languages: List[str]
    default_k: int
    default_temperature: float
    refusal_threshold: float
    min_relevance: float
    rate_limit_rps: float
    rate_limit_burst: int
    app_env: str


class DebugPromptResponse(BaseModel):
    prompt: str
    retrieved: List[str]
    lang: str
    refusal: bool
    reason: Optional[str]
