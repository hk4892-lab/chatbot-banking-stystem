"""FastAPI application wiring for BankBot."""

from __future__ import annotations

import importlib.util
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field

from .core.language import Language, available_languages, combined_language, detect_language
from .core.policy import ConversationState, ConversationTurn, DialoguePolicy
from .core.redaction import PIIRedactor, tokens_only
from .core.retrieval import RetrievalEngine

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "app" / "data"
LOG_PATH = APP_ROOT / "logs" / "audit.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

SENTENCE_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None

BASE_RETRIEVER = RetrievalEngine(DATA_DIR, use_sentence_transformers=False)
RETRIEVER_CACHE: dict[str, RetrievalEngine] = {
    "tfidf": BASE_RETRIEVER,
    "st": None,
}

policy = DialoguePolicy(BASE_RETRIEVER)
redactor = PIIRedactor()


class SessionStore:
    """In-memory store for short-lived sessions."""

    def __init__(self) -> None:
        self._data: dict[str, ConversationState] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> ConversationState:
        with self._lock:
            if session_id not in self._data:
                self._data[session_id] = ConversationState()
            return self._data[session_id]


sessions = SessionStore()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    threshold: float | None = None
    lang: Language | None = None
    use_sentence_transformers: bool | None = None
    escalate: bool = False

    class Config:
        use_enum_values = True


class ChatResponse(BaseModel):
    lang: str
    route: str
    answer: str | None
    citations: list[dict[str, str]]
    tool_result: dict | None
    confidence: float


app = FastAPI(title="BankBot", version="0.1.0")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
async def config() -> dict[str, Any]:
    return {
        "threshold": policy.default_threshold,
        "languages": [lang.value for lang in available_languages()],
        "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
        "kb_entries": len(BASE_RETRIEVER.entries),
    }


@app.post("/chat/turn", response_model=ChatResponse)
async def chat_turn(payload: ChatRequest, response: Response, request: Request) -> ChatResponse:
    session_id = payload.session_id or str(uuid4())
    response.headers["X-Session-ID"] = session_id
    session = sessions.get(session_id)

    detected = detect_language(payload.message)
    resolved_lang = combined_language(payload.lang, detected)

    redacted_message, token_map = redactor.redact(payload.message)
    if token_map:
        session.token_map.update(token_map)

    session.remember(ConversationTurn(role="user", text=redacted_message))

    retriever = _select_retriever(payload.use_sentence_transformers)
    policy.retriever = retriever

    if payload.escalate:
        decision = policy.decide("", resolved_lang, session, threshold=payload.threshold)
        decision.route = "escalate"
        decision.answer = (
            "A human support specialist will reach out shortly. "
            "Feel free to share a reachable contact number if comfortable."
        )
        if session.turns:
            session.turns[-1].text = decision.answer
    else:
        decision = policy.decide(redacted_message, resolved_lang, session, threshold=payload.threshold)

    _write_audit_log(
        {
            "ts": datetime.utcnow().isoformat() + "Z",
            "session": session_id,
            "lang": resolved_lang.value,
            "route": decision.route,
            "topdoc_id": decision.top_doc_id,
            "top_score": round(decision.top_score, 4),
            "confidence": round(decision.confidence, 4),
            "tools_called": decision.tools_called,
            "redaction_tokens": tokens_only(token_map),
            "ip": request.client.host if request.client else None,
        }
    )

    return ChatResponse(
        lang=resolved_lang.value,
        route=decision.route,
        answer=decision.answer,
        citations=decision.citations,
        tool_result=decision.tool_result,
        confidence=decision.confidence,
    )


def _select_retriever(use_st_flag: bool | None) -> RetrievalEngine:
    if not use_st_flag:
        return RETRIEVER_CACHE["tfidf"]
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return RETRIEVER_CACHE["tfidf"]
    if RETRIEVER_CACHE.get("st") is None:
        RETRIEVER_CACHE["st"] = RetrievalEngine(DATA_DIR, use_sentence_transformers=True)
    return RETRIEVER_CACHE["st"]


def _write_audit_log(payload: Dict[str, Any]) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
