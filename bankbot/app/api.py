"""FastAPI application entrypoint for BankBot."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.core.language import LanguageDetection, detect_language
from app.core.policy import PolicyResult, PolicyRouter, ToolExecutor
from app.core.redaction import redact_text
from app.core.retrieval import Retriever, SENTENCE_TRANSFORMERS_AVAILABLE
from app.tools import tools as tool_impl


APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "app" / "data"
LOG_PATH = APP_ROOT / "logs" / "audit.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


class AuditLogger:
    """Write structured JSONL audit logs for each turn."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, payload: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


class ChatTurnRequest(BaseModel):
    message: str = Field(..., min_length=1)
    lang: str = Field("AUTO", description="Override language routing")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    use_sentence_transformers: bool = Field(False, description="Enable sentence-transformers reranker")


class ChatTurnResponse(BaseModel):
    lang: str
    route: str
    answer: Optional[str]
    citations: List[Dict[str, str]]
    tool_result: Optional[Dict[str, Any]]
    confidence: float


class SessionMemory:
    def __init__(self, max_turns: int = 20) -> None:
        self.turns: Deque[Dict[str, Any]] = deque(maxlen=max_turns)
        self.tokens: Dict[str, str] = {}

    def record(self, role: str, content: Dict[str, Any]) -> None:
        self.turns.append({"role": role, **content})

    def update_tokens(self, token_map: Dict[str, str]) -> None:
        self.tokens.update(token_map)


app = FastAPI(title="BankBot", version="0.1.0")

retriever = Retriever(DATA_DIR)
tool_executor = ToolExecutor(tool_impl)
policy_router = PolicyRouter(retriever=retriever, tool_executor=tool_executor)
audit_logger = AuditLogger(LOG_PATH)
session_memory = SessionMemory()


def _resolve_lang_override(lang_value: str, detected: LanguageDetection) -> LanguageDetection:
    lang_value = (lang_value or "AUTO").upper()
    if lang_value == "AUTO":
        return detected
    if lang_value not in {"EN", "HI", "TA"}:
        raise HTTPException(status_code=400, detail="Unsupported language override")
    return LanguageDetection(primary=lang_value, detected_scripts=[lang_value], is_mixed=False)


def _default_threshold(request_threshold: Optional[float]) -> float:
    if request_threshold is not None:
        return request_threshold
    return retriever.threshold


def _log_turn(
    result: PolicyResult,
    redaction_tokens: List[str],
    threshold: float,
    use_sentence_transformers: bool,
    sanitized_message: str,
) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "lang": result.lang,
        "route": result.route,
        "topdoc_id": result.top_doc_id,
        "top_score": round(result.confidence, 4),
        "redaction_tokens": sorted(redaction_tokens),
        "tools_called": [result.tool_result.get("tool")] if result.tool_result else [],
        "confidence_threshold": threshold,
        "sentence_transformers": use_sentence_transformers,
        "message_hash": hashlib.sha1(sanitized_message.encode("utf-8")).hexdigest(),
    }
    audit_logger.log(payload)


@app.get("/healthz")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def config_snapshot() -> Dict[str, Any]:
    return {
        "languages": ["EN", "HI", "TA"],
        "default_threshold": retriever.threshold,
        "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
    }


@app.post("/chat/turn", response_model=ChatTurnResponse)
def chat_turn(request: ChatTurnRequest) -> ChatTurnResponse:
    redaction_result = redact_text(request.message)
    session_memory.update_tokens(redaction_result.token_map)

    detected = detect_language(request.message)
    lang_detection = _resolve_lang_override(request.lang, detected)

    threshold = _default_threshold(request.confidence_threshold)
    use_sentence_transformers = bool(
        request.use_sentence_transformers and SENTENCE_TRANSFORMERS_AVAILABLE
    )

    result = policy_router.handle(
        message=redaction_result.sanitized_text,
        lang_detection=lang_detection,
        confidence_threshold=threshold,
        use_sentence_transformers=use_sentence_transformers,
        token_map=session_memory.tokens,
    )

    session_memory.record("user", {"message": redaction_result.sanitized_text})
    session_memory.record("assistant", {"route": result.route, "message": result.answer or ""})

    _log_turn(
        result=result,
        redaction_tokens=list(redaction_result.token_map.keys()),
        threshold=threshold,
        use_sentence_transformers=use_sentence_transformers,
        sanitized_message=redaction_result.sanitized_text,
    )

    response = ChatTurnResponse(
        lang=lang_detection.primary,
        route=result.route,
        answer=result.answer,
        citations=result.citations,
        tool_result=result.tool_result,
        confidence=result.confidence,
    )
    return response
