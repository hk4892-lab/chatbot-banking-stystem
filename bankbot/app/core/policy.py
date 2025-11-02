"""Dialogue policy for routing and tool invocation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .language import Language, normalize_text
from .redaction import PIIRedactor, TokenMap
from .render import answer_from_results, clarifying_prompt, citations_from_results
from .retrieval import RetrievalEngine
from ..tools import tools as toolset


@dataclass
class ConversationTurn:
    role: str
    text: str
    route: str | None = None
    confidence: float | None = None


@dataclass
class ConversationState:
    turns: list[ConversationTurn] = field(default_factory=list)
    token_map: TokenMap = field(default_factory=dict)

    def remember(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)


@dataclass
class PolicyDecision:
    route: str
    answer: str | None
    citations: list[dict[str, str]]
    tool_result: dict | None
    confidence: float
    tools_called: list[str] = field(default_factory=list)
    top_doc_id: str | None = None
    top_score: float = 0.0


class DialoguePolicy:
    """Rule-based dialogue policy for BankBot."""

    def __init__(self, retriever: RetrievalEngine, default_threshold: float = 0.18) -> None:
        self.retriever = retriever
        self.default_threshold = default_threshold

    def decide(
        self,
        message: str,
        lang: Language,
        session: ConversationState,
        threshold: float | None = None,
    ) -> PolicyDecision:
        threshold = threshold if threshold is not None else self.default_threshold

        tool_decision = self._maybe_handle_tool(message, session)
        if tool_decision:
            session.remember(
                ConversationTurn(role="bot", text=tool_decision.answer or "", route="tool", confidence=1.0)
            )
            return tool_decision

        results = self.retriever.search(message)
        top_score = results[0].score if results else 0.0
        top_doc_id = results[0].entry.id if results else None

        if not results or top_score < threshold * 0.5:
            refusal = (
                "Sorry, I'm not confident enough to answer that. "
                "Let me escalate you to a human specialist if you need immediate support."
            )
            decision = PolicyDecision(
                route="escalate",
                answer=refusal,
                citations=[],
                tool_result=None,
                confidence=top_score,
                tools_called=[],
                top_doc_id=top_doc_id,
                top_score=top_score,
            )
            session.remember(ConversationTurn(role="bot", text=refusal, route="escalate", confidence=top_score))
            return decision

        if top_score < threshold:
            question, suggestions = clarifying_prompt(results[0])
            answer = question
            if suggestions:
                bullet_lines = "\n".join(f"- {item}" for item in suggestions)
                answer = f"{question}\n\nHere are a couple of ways to rephrase:\n{bullet_lines}"
            decision = PolicyDecision(
                route="ask",
                answer=answer,
                citations=citations_from_results(results[:1]),
                tool_result=None,
                confidence=top_score,
                tools_called=[],
                top_doc_id=top_doc_id,
                top_score=top_score,
            )
            session.remember(ConversationTurn(role="bot", text=answer, route="ask", confidence=top_score))
            return decision

        answer_text = answer_from_results(results)
        decision = PolicyDecision(
            route="answer",
            answer=answer_text,
            citations=citations_from_results(results),
            tool_result=None,
            confidence=top_score,
            tools_called=[],
            top_doc_id=top_doc_id,
            top_score=top_score,
        )
        session.remember(ConversationTurn(role="bot", text=answer_text, route="answer", confidence=top_score))
        return decision

    def _maybe_handle_tool(self, message: str, session: ConversationState) -> PolicyDecision | None:
        normalized = normalize_text(message)

        emi_request = _parse_emi_request(normalized)
        if emi_request:
            emi = toolset.calculate_emi(
                P=emi_request["principal"],
                annual_rate_percent=emi_request["rate"],
                months=emi_request["months"],
            )
            answer = (
                f"The estimated EMI is ?{emi:,.2f} for a principal of ?{emi_request['principal']:,.0f} "
                f"at {emi_request['rate']}% for {emi_request['months']} months."
            )
            decision = PolicyDecision(
                route="tool",
                answer=answer,
                citations=[],
                tool_result={"tool": "calculate_emi", "emi": round(emi, 2)},
                confidence=1.0,
                tools_called=["calculate_emi"],
            )
            return decision

        if _is_block_card_request(normalized):
            card_token = _first_token(session.token_map, prefix="[[CARD_")
            reason = "suspicious activity"
            if card_token:
                card_number = PIIRedactor.detokenize(card_token, session.token_map)
                ticket = toolset.block_card(card_number, reason)
                answer = (
                    "I have raised a block request for the mentioned card. "
                    f"Your reference id is {ticket}."
                )
                return PolicyDecision(
                    route="tool",
                    answer=answer,
                    citations=[],
                    tool_result={"tool": "block_card", "ticket": ticket, "token": card_token},
                    confidence=1.0,
                    tools_called=["block_card"],
                )
            answer = (
                "I can help with card blocking. Please provide the card number so I can redact and process it."
            )
            return PolicyDecision(
                route="ask",
                answer=answer,
                citations=[],
                tool_result=None,
                confidence=0.4,
                tools_called=[],
            )

        if "rate" in normalized or "interest" in normalized:
            product = _extract_product(normalized)
            if product:
                rate_info = toolset.fetch_rate(product)
                answer = (
                    f"The current {rate_info['product']} rate is {rate_info['rate_percent']}%. "
                    "These rates are indicative for demo purposes."
                )
                return PolicyDecision(
                    route="tool",
                    answer=answer,
                    citations=[],
                    tool_result=rate_info,
                    confidence=0.9,
                    tools_called=["fetch_rate"],
                )

        return None


def _parse_emi_request(text: str) -> dict[str, float] | None:
    if "emi" not in text:
        return None

    explicit_pattern = re.compile(
        r"emi.*?p\s*=?\s*([\d.,kmlakh]+).*?r\s*=?\s*([\d.,]+).*?(?:n|months?)\s*=?\s*([\d]+)",
        re.IGNORECASE,
    )
    match = explicit_pattern.search(text)
    if match:
        principal = _parse_amount(match.group(1))
        rate = float(match.group(2).replace("%", ""))
        months = int(match.group(3))
        if principal and rate and months:
            return {"principal": principal, "rate": rate, "months": months}

    values = [token.replace("%", "") for token in re.findall(r"[\d.]+%?", text)]
    numeric = [token for token in values if token]

    if len(numeric) >= 3:
        principal = _parse_amount(numeric[0])
        rate = float(numeric[1])
        months = int(float(numeric[2]))
        if principal and rate and months:
            return {"principal": principal, "rate": rate, "months": months}

    return None


def _parse_amount(raw: str) -> float:
    cleaned = raw.lower().replace(",", "").strip()
    multiplier = 1
    for suffix, factor in (("lakh", 100000), ("lac", 100000), ("l", 100000), ("k", 1000)):
        if cleaned.endswith(suffix):
            multiplier = factor
            cleaned = cleaned[: -len(suffix)]
            break
    try:
        value = float(cleaned)
    except ValueError:
        return 0.0
    return value * multiplier


def _is_block_card_request(text: str) -> bool:
    return "block" in text and "card" in text


def _extract_product(text: str) -> str | None:
    products = {
        "savings": "savings account",
        "fd": "fixed deposit",
        "fixed deposit": "fixed deposit",
        "home loan": "home loan",
        "loan": "loan",
        "credit card": "credit card",
    }
    for key, value in products.items():
        if key in text:
            return value
    return None


def _first_token(mapping: TokenMap, prefix: str) -> str | None:
    for token in mapping:
        if token.startswith(prefix):
            return token
    return None
