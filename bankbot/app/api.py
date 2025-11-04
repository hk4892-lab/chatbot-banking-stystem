"""FastAPI application exposing the BankBot chatbot endpoints."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import EMBED_MODEL, LOG_PATH, TOP_K, USE_SLM
from .language import detect_language
from .policy import PolicyDecision, decide_action
from .rag import build_or_load_index, index_ready, retrieve
from .redaction import safe_for_log
from .slm import generate as slm_generate
from .slm import slm_available
from .tools import block_card_ticket, emi_calculator, interest_rates


LOGGER = logging.getLogger("bankbot.api")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%Y-%m-%d %H:%M:%S | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: List[Message]
    top_k: int = Field(TOP_K, ge=1, le=8)
    max_new_tokens: int = Field(256, ge=64, le=512)
    temperature: float = Field(0.2, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    reply: str
    lang: Literal["en", "ta", "hi"]
    source: Literal["tool", "rag", "slm_rag", "clarify", "escalate"]


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
    rag_index_ready = index_ready() or build_or_load_index().is_ready
    return {
        "ok": True,
        "rag_index": rag_index_ready,
        "slm": bool(USE_SLM and slm_available()),
        "embed_model": EMBED_MODEL,
    }


def _latest_user(messages: List[Message]) -> Message:
    for item in reversed(messages):
        if item.role == "user":
            return item
    raise HTTPException(status_code=400, detail="At least one user message is required")


def _lang_strings(lang: str) -> Dict[str, str]:
    table = {
        "en": {
            "clarify": "Could you share more details so I can assist accurately?",
            "escalate": "I’m unable to process that request securely. Please contact our official support team.",
            "intent_label": "Intent",
            "rag_additional": "Additional context: {context}",
            "rag_other_intents": "Related intents: {intents}",
            "emi": "The estimated EMI is {emi} with a total payment of {total_payment} and interest of {total_interest}.",
            "emi_suffix": "Please review the schedule before confirming.",
            "card": "A block request for the card ending {last4} is raised. Ticket ID: {ticket_id}.",
            "card_suffix": "Our support team will call you shortly to verify.",
            "interest": "The current rate for {product} is {rate_percent:.1f}% per annum.",
            "interest_suffix": "Rates may change, so confirm before booking.",
        },
        "ta": {
            "clarify": "சரியாக உதவ மேலும் விவரங்களை வழங்கவும்.",
            "escalate": "இந்த கோரிக்கையை பாதுகாப்பாக நிறைவேற்ற முடியவில்லை. தயவுசெய்து அதிகாரப்பூர்வ வங்கி உதவியை தொடர்பு கொள்ளவும்.",
            "intent_label": "நோக்கம்",
            "rag_additional": "கூடுதல் தகவல்: {context}",
            "rag_other_intents": "தொடர்பான நோக்கங்கள்: {intents}",
            "emi": "கணிக்கப்பட்ட EMI {emi}, மொத்த கட்டணம் {total_payment}, வட்டி {total_interest} ஆகும்.",
            "emi_suffix": "உறுதிப்படுத்தும் முன் திட்டத்தை சரிபார்க்கவும்.",
            "card": "{last4} இலக்கத்தில் முடியும் கார்டிற்கான தடுப்பு கோரிக்கை பதிவு செய்யப்பட்டது. குறிப்பு: {ticket_id}.",
            "card_suffix": "உதவி குழு விரைவில் தொடர்பு கொள்கிறது.",
            "interest": "{product} பற்றிய தற்போதைய வட்டி {rate_percent:.1f}% வருடத்திற்கு.",
            "interest_suffix": "விகிதம் மாற்றமடையலாம்; பதிவு செய்வதற்கு முன் உறுதிப்படுத்தவும்.",
        },
        "hi": {
            "clarify": "कृपया सही सहायता के लिए कुछ और विवरण बताएँ।",
            "escalate": "मैं इस अनुरोध को सुरक्षित रूप से पूरा नहीं कर सकता। कृपया हमारे आधिकारिक सहायता केंद्र से संपर्क करें।",
            "intent_label": "इंटेंट",
            "rag_additional": "अतिरिक्त संदर्भ: {context}",
            "rag_other_intents": "संबंधित इंटेंट: {intents}",
            "emi": "अनुमानित ईएमआई {emi} है, कुल भुगतान {total_payment} और ब्याज {total_interest} होगा.",
            "emi_suffix": "कृपया अनुसूची की पुष्टि से पहले जाँच लें.",
            "card": "{last4} पर समाप्त होने वाले कार्ड के लिए ब्लॉक अनुरोध दर्ज कर दिया गया है। टिकट आईडी: {ticket_id}.",
            "card_suffix": "हमारी टीम शीघ्र ही पुष्टि के लिए कॉल करेगी.",
            "interest": "{product} के लिए वर्तमान दर {rate_percent:.1f}% प्रति वर्ष है.",
            "interest_suffix": "दर बदल सकती है; बुकिंग से पहले सत्यापित करें.",
        },
    }
    return table.get(lang, table["en"])


def _format_currency(value: float, *, decimals: int = 2) -> str:
    formatted = f"{value:,.{decimals}f}"
    return f"₹{formatted}"


def _tool_response(tool: str, message: str, lang: str) -> str:
    strings = _lang_strings(lang)
    lowered = message.lower()

    if tool == "emi_calculator":
        numbers = [float(num) for num in re.findall(r"\d+(?:\.\d+)?", message)]
        principal = numbers[0] if numbers else 100000.0
        annual_rate = numbers[1] if len(numbers) > 1 else 8.5
        tenure = int(numbers[2]) if len(numbers) > 2 else 60
        result = emi_calculator(principal, annual_rate, tenure)
        emi = _format_currency(result["emi"])
        total_payment = _format_currency(result["total_payment"])
        total_interest = _format_currency(result["total_interest"])
        return " ".join(
            [
                strings["emi"].format(emi=emi, total_payment=total_payment, total_interest=total_interest),
                strings["emi_suffix"],
            ]
        )

    if tool == "block_card_ticket":
        last4_match = re.search(r"(\d{4})\b", message)
        last4 = last4_match.group(1) if last4_match else "0000"
        ticket = block_card_ticket("Customer", last4, message)
        return " ".join(
            [
                strings["card"].format(last4=last4, ticket_id=ticket["ticket_id"]),
                strings["card_suffix"],
            ]
        )

    if tool == "interest_rates":
        product_map = {
            "savings": {"savings", "savings account"},
            "fixed deposit": {"fixed deposit", "fd"},
            "home loan": {"home loan", "housing loan"},
            "personal loan": {"personal loan"},
        }
        product = "savings"
        for key, variants in product_map.items():
            if any(term in lowered for term in variants):
                product = key
                break
        rate_info = interest_rates(product.replace(" ", "_"))
        clean_product = product.title()
        return " ".join(
            [
                strings["interest"].format(product=clean_product, rate_percent=rate_info["rate_percent"]),
                strings["interest_suffix"],
            ]
        )

    return strings["clarify"]


def _language_name(code: str) -> str:
    return {"en": "English", "ta": "Tamil", "hi": "Hindi"}.get(code, "English")


def _rag_response(hits: List[Dict], lang: str) -> str:
    if not hits:
        return ""
    strings = _lang_strings(lang)
    primary = hits[0]
    intent_label = strings["intent_label"]
    reply_parts = [f"{primary['text']} ({intent_label}: {primary['intent'] or 'general'})"]

    extra_contexts = [hit["text"] for hit in hits[1:] if hit.get("text")]
    if extra_contexts:
        reply_parts.append(strings["rag_additional"].format(context=extra_contexts[0]))

    other_intents = {hit["intent"] for hit in hits if hit.get("intent")}
    if other_intents:
        reply_parts.append(strings["rag_other_intents"].format(intents=", ".join(sorted(other_intents))))

    return " ".join(reply_parts[:3])


def _slm_context_system(lang: str, hits: List[Dict]) -> str:
    language_name = _language_name(lang)
    context_lines = []
    for hit in hits:
        intent = hit.get("intent", "") or "general"
        snippet = hit.get("context", "")[:350]
        context_lines.append(f"- ({intent}) {snippet}")
    context_block = "\n".join(context_lines) if context_lines else "- No context"
    return (
        "You are a multilingual banking assistant. Use only the provided context to answer. "
        "Do not fabricate account or transaction details. Keep responses concise (2-5 sentences) "
        f"and reply in {language_name}.\n\nContext:\n{context_block}"
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")

    latest = _latest_user(request.messages)
    lang = detect_language(latest.content)
    LOGGER.info("chat.request lang=%s text=%s", lang, safe_for_log(latest.content))

    top_k = min(max(request.top_k, 1), 8)
    hits = retrieve(latest.content, top_k)
    best_score = hits[0]["score"] if hits else 0.0
    top_intent = hits[0].get("intent") if hits else None
    slm_enabled = bool(USE_SLM and slm_available())

    decision: PolicyDecision = decide_action(
        latest.content,
        retrieval_score=best_score,
        use_slm=slm_enabled,
        top_intent=top_intent,
    )

    if decision.action == "escalate":
        reply = _lang_strings(lang)["escalate"]
        LOGGER.info("chat.response source=escalate text=%s", safe_for_log(reply))
        return ChatResponse(reply=reply, lang=lang, source="escalate")

    if decision.action == "tool" and decision.tool:
        reply = _tool_response(decision.tool, latest.content, lang)
        LOGGER.info("chat.response source=tool text=%s", safe_for_log(reply))
        return ChatResponse(reply=reply, lang=lang, source="tool")

    if decision.action == "rag" and hits:
        reply = _rag_response(hits, lang)
        LOGGER.info(
            "chat.response source=rag score=%.4f text=%s",
            best_score,
            safe_for_log(reply),
        )
        return ChatResponse(reply=reply, lang=lang, source="rag")

    if decision.action == "slm_rag" and slm_enabled:
        system_prompt = _slm_context_system(lang, hits)
        slm_reply = slm_generate(
            system_prompt,
            [message.model_dump() for message in request.messages],
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        if slm_reply:
            LOGGER.info("chat.response source=slm_rag text=%s", safe_for_log(slm_reply))
            return ChatResponse(reply=slm_reply, lang=lang, source="slm_rag")

    reply = _lang_strings(lang)["clarify"]
    LOGGER.info("chat.response source=clarify text=%s", safe_for_log(reply))
    return ChatResponse(reply=reply, lang=lang, source="clarify")


__all__ = ["app"]
