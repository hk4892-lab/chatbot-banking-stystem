"""Conversation policy and routing decisions for BankBot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .retrieval import SCORE_THRESHOLD


SAFE_ESCALATION_MESSAGE = (
    "I’m unable to help with that securely. Please contact our official support channel for further assistance."
)

_SENSITIVE_KEYWORDS = [
    "pin",
    "password",
    "otp",
    "cvv",
    "one time password",
    "login code",
]

_IRREVERSIBLE_PATTERNS = [
    "close my account",
    "permanently delete",
    "transfer all money",
]

_EMI_KEYWORDS = ["emi", "installment", "monthly payment", "loan repayment"]
_BLOCK_CARD_KEYWORDS = ["block", "lost card", "stolen card", "card freeze"]
_INTEREST_KEYWORDS = ["interest rate", "rate of interest", "roi", "interest"]


@dataclass
class PolicyDecision:
    action: str
    tool: Optional[str] = None
    strategy: Optional[str] = None
    message: Optional[str] = None


def _contains_keywords(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def decide_action(
    user_text: str,
    retrieval_score: float,
    *,
    use_slm: bool,
) -> PolicyDecision:
    """Determine whether to answer, clarify, escalate, or invoke a tool."""

    lowered = user_text.lower()

    if _contains_keywords(lowered, _SENSITIVE_KEYWORDS) or _contains_keywords(
        lowered, _IRREVERSIBLE_PATTERNS
    ):
        return PolicyDecision(action="escalate", message=SAFE_ESCALATION_MESSAGE)

    if _contains_keywords(lowered, _EMI_KEYWORDS):
        return PolicyDecision(action="tool", tool="emi_calculator")
    if "block" in lowered and "card" in lowered or _contains_keywords(
        lowered, _BLOCK_CARD_KEYWORDS
    ):
        return PolicyDecision(action="tool", tool="block_card_ticket")
    if _contains_keywords(lowered, _INTEREST_KEYWORDS):
        return PolicyDecision(action="tool", tool="interest_rates")

    if retrieval_score >= SCORE_THRESHOLD:
        return PolicyDecision(action="answer", strategy="rag")

    if use_slm:
        return PolicyDecision(action="answer", strategy="slm")

    return PolicyDecision(action="ask_clarify")


__all__ = ["PolicyDecision", "decide_action", "SAFE_ESCALATION_MESSAGE"]
