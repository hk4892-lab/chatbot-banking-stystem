"""Conversation policy for BankBot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import RAG_SCORE_THRESHOLD


SENSITIVE_KEYWORDS = {
    "pin",
    "password",
    "otp",
    "cvv",
    "one time password",
    "login code",
    "kyc document",
    "upload id",
}

IRREVERSIBLE_PHRASES = {
    "close my account",
    "permanently delete",
    "transfer all money",
    "sell all holdings",
}

TOOL_KEYWORDS = {
    "emi_calculator": {"emi", "installment", "monthly payment", "loan calculator"},
    "block_card_ticket": {"block card", "card lost", "card stolen", "freeze card"},
    "interest_rates": {"interest rate", "loan rate", "fixed deposit rate", "roi"},
}

INTENT_TO_TOOL = {
    "card_services": "block_card_ticket",
    "loan_services": "emi_calculator",
    "balance_inquiry": None,
}


@dataclass
class PolicyDecision:
    action: str
    tool: Optional[str] = None


def _contains(text: str, phrases: set[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _detect_tool(text: str, intent: Optional[str]) -> Optional[str]:
    lowered = text.lower()
    for tool, keywords in TOOL_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return tool
    if intent and intent in INTENT_TO_TOOL:
        return INTENT_TO_TOOL[intent]
    return None


def decide_action(
    message: str,
    *,
    retrieval_score: float,
    use_slm: bool,
    top_intent: Optional[str] = None,
) -> PolicyDecision:
    lowered = message.lower()

    if _contains(lowered, SENSITIVE_KEYWORDS) or _contains(lowered, IRREVERSIBLE_PHRASES):
        return PolicyDecision(action="escalate")

    tool = _detect_tool(message, top_intent)
    if tool:
        return PolicyDecision(action="tool", tool=tool)

    if retrieval_score >= RAG_SCORE_THRESHOLD:
        return PolicyDecision(action="rag")

    if use_slm:
        return PolicyDecision(action="slm_rag")

    return PolicyDecision(action="clarify")


__all__ = ["PolicyDecision", "decide_action"]
