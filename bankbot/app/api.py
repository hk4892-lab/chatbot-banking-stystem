"""FastAPI application exposing the BankBot chatbot endpoints."""

from __future__ import annotations

import logging
import re
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import LOG_PATH, USE_SLM
from .language import detect_language
from .policy import SAFE_ESCALATION_MESSAGE, PolicyDecision, decide_action
from .redaction import redact_for_logs
from .retrieval import is_ready as retrieval_ready
from .retrieval import retrieve
from .slm_module import slm_available, slm_generate
from .tools import block_card_ticket, emi_calculator, interest_rates


LOGGER = logging.getLogger("bankbot")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: int = Field(300, ge=32, le=512)
    temperature: float = Field(0.2, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    reply: str
    lang: Literal["en", "ta", "hi"]
    source: Literal["rag", "slm", "tool", "clarify", "escalate"]


app = FastAPI(title="BankBot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"ok": True, "msg": "BankBot API up. See /docs"}


@app.get("/healthz")
def healthcheck():
    slm_status = bool(USE_SLM and slm_available())
    return {"ok": True, "rag": retrieval_ready(), "slm": slm_status}


def _get_latest_user_message(messages: List[Message]) -> Message:
    for message in reversed(messages):
        if message.role == "user":
            return message
    raise HTTPException(status_code=400, detail="At least one user message is required")


def _collect_system_prompt(messages: List[Message]) -> str:
    system_messages = [msg.content for msg in messages if msg.role == "system"]
    if system_messages:
        return "\n".join(system_messages)
    return (
        "You are BankBot, a multilingual banking assistant that responds concisely, "
        "follows policy, and never fabricates irreversible actions."
    )


def _extract_numbers(text: str) -> List[float]:
    candidates = re.findall(r"\d+(?:\.\d+)?", text)
    return [float(c) for c in candidates]


def _handle_tool(decision: PolicyDecision, message: Message) -> ChatResponse:
    lowered = message.content.lower()
    lang = detect_language(message.content)

    if decision.tool == "emi_calculator":
        numbers = _extract_numbers(message.content)
        principal = numbers[0] if numbers else 100000.0
        annual_rate = numbers[1] if len(numbers) > 1 else 8.5
        tenure = int(numbers[2]) if len(numbers) > 2 else 60
        result = emi_calculator(principal, annual_rate, tenure)
        reply = (
            f"Based on a principal of ₹{principal:,.0f}, the EMI is approximately ₹{result['emi']:,.2f}. "
            f"Total payment is ₹{result['total_payment']:,.2f} with interest of ₹{result['total_interest']:,.2f}."
        )
        return ChatResponse(reply=reply, lang=lang, source="tool")

    if decision.tool == "block_card_ticket":
        last4_match = re.search(r"(\d{4})\b", message.content)
        last4 = last4_match.group(1) if last4_match else "0000"
        result = block_card_ticket("Customer", last4, message.content)
        reply = (
            "I have raised a temporary block ticket for your card ending "
            f"{last4}. Reference ID: {result['ticket_id']}. Our team will call you shortly."
        )
        return ChatResponse(reply=reply, lang=lang, source="tool")

    if decision.tool == "interest_rates":
        product_match = re.search(r"(savings|fixed deposit|home loan|personal loan)", lowered)
        product = product_match.group(1) if product_match else "savings"
        result = interest_rates(product)
        reply = (
            f"The current interest rate for {product.replace('_', ' ')} is {result['rate_percent']:.1f}% per annum."
        )
        return ChatResponse(reply=reply, lang=lang, source="tool")

    return ChatResponse(
        reply="I can help once I know which banking tool to use. Could you clarify your request?",
        lang=lang,
        source="clarify",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")

    latest_user = _get_latest_user_message(request.messages)
    lang = detect_language(latest_user.content)
    redacted = redact_for_logs(latest_user.content)
    LOGGER.info("chat_request lang=%s text=%s", lang, redacted)

    retrieval_result = retrieve(latest_user.content)
    slm_ok = USE_SLM and slm_available()
    decision = decide_action(latest_user.content, retrieval_result.get("score", 0.0), use_slm=slm_ok)

    if decision.action == "escalate":
        return ChatResponse(reply=decision.message or SAFE_ESCALATION_MESSAGE, lang=lang, source="escalate")

    if decision.action == "tool":
        return _handle_tool(decision, latest_user)

    if decision.action == "answer" and decision.strategy == "rag" and retrieval_result.get("answer"):
        reply = retrieval_result["answer"]
        return ChatResponse(reply=reply, lang=retrieval_result.get("lang", lang), source="rag")

    if decision.action == "answer" and decision.strategy == "slm" and slm_ok:
        system_prompt = _collect_system_prompt(request.messages)
        slm_reply = slm_generate(system_prompt, [msg.model_dump() for msg in request.messages])
        if slm_reply:
            return ChatResponse(reply=slm_reply, lang=lang, source="slm")

    if retrieval_result.get("answer"):
        reply = (
            "I have some information that might help: "
            f"{retrieval_result['answer']}"
        )
        return ChatResponse(reply=reply, lang=retrieval_result.get("lang", lang), source="rag")

    clarify_text = "Could you share a few more details so I can assist accurately?"
    return ChatResponse(reply=clarify_text, lang=lang, source="clarify")


__all__ = ["app"]
