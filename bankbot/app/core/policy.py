"""Routing policy deciding how to respond to user messages."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple

from .language import LanguageDetection
from .render import render_answer
from .retrieval import Retriever, SearchResponse, format_citations


@dataclass
class PolicyResult:
    lang: str
    route: str
    answer: Optional[str]
    citations: List[Dict[str, str]]
    tool_result: Optional[Dict[str, Any]]
    confidence: float
    top_doc_id: Optional[str]


@dataclass
class ToolIntent:
    name: str
    args: Dict[str, Any]


EMI_TRIGGER = re.compile(r"emi", re.IGNORECASE)
UNIT_MULTIPLIERS: Dict[str, float] = {
    "l": 1e5,
    "lac": 1e5,
    "lakh": 1e5,
    "cr": 1e7,
    "crore": 1e7,
}


def _parse_amount(value: str, unit: Optional[str]) -> float:
    amount = float(value.replace(",", ""))
    if unit:
        amount *= UNIT_MULTIPLIERS.get(unit.lower(), 1.0)
    return amount


def _extract_principal(message: str) -> Optional[float]:
    match = re.search(r"p\s*=\s*([\d.,]+)\s*(l|lac|lakh|cr|crore)?", message, re.IGNORECASE)
    if match:
        return _parse_amount(match.group(1), match.group(2))
    return None


def _extract_rate(message: str) -> Optional[float]:
    match = re.search(r"(?:r|rate)\s*=\s*([\d.,]+)", message, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))
    percent = re.search(r"([\d.,]+)\s*%", message)
    if percent:
        return float(percent.group(1).replace(",", ""))
    return None


def _extract_months(message: str) -> Optional[int]:
    match = re.search(r"(?:n|tenure)\s*=\s*(\d+)", message, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\s*(?:months|month|m)\b", message, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


NUMBER_WITH_UNITS = re.compile(r"(\d+(?:[.,]\d+)?)(?:\s*(l|lac|lakh|cr|crore))?", re.IGNORECASE)


def _fallback_emi_values(message: str) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    matches = list(NUMBER_WITH_UNITS.finditer(message))
    if len(matches) >= 3:
        principal_match = matches[0]
        rate_match = matches[1]
        months_match = matches[2]
        principal = _parse_amount(principal_match.group(1), principal_match.group(2))
        rate = float(rate_match.group(1).replace(",", ""))
        months = int(float(months_match.group(1).replace(",", "")))
        return principal, rate, months
    return None, None, None

BLOCK_TRIGGER = re.compile(r"block\s+.*card", re.IGNORECASE)
RATE_TRIGGER = re.compile(r"rate", re.IGNORECASE)


class PolicyRouter:
    """Deterministic router that picks between answer/tool/ask/escalate."""

    def __init__(self, retriever: Retriever, tool_executor: "ToolExecutor") -> None:
        self.retriever = retriever
        self.tool_executor = tool_executor

    def _detect_tool(self, message: str) -> Optional[ToolIntent]:
        if EMI_TRIGGER.search(message):
            principal = _extract_principal(message)
            rate = _extract_rate(message)
            months = _extract_months(message)

            if principal is None or rate is None or months is None:
                fallback_principal, fallback_rate, fallback_months = _fallback_emi_values(message)
                principal = principal or fallback_principal
                rate = rate or fallback_rate
                months = months or fallback_months

            if principal and rate and months:
                return ToolIntent(
                    name="calculate_emi",
                    args={
                        "P": principal,
                        "annual_rate_percent": rate,
                        "months": int(months),
                    },
                )

        if BLOCK_TRIGGER.search(message):
            reason_match = re.search(r"because\s+(.*)$", message, re.IGNORECASE)
            reason = reason_match.group(1).strip() if reason_match else "customer request"
            return ToolIntent(name="block_card", args={"tokenized_card": "", "reason": reason})

        if RATE_TRIGGER.search(message):
            product_match = re.search(
                r"(fd|fixed deposit|savings|home loan|education loan|personal loan)",
                message,
                re.IGNORECASE,
            )
            if product_match:
                product = product_match.group(1).lower()
                return ToolIntent(name="fetch_rate", args={"product": product})

        return None

    def handle(
        self,
        message: str,
        lang_detection: LanguageDetection,
        confidence_threshold: float,
        use_sentence_transformers: bool,
        token_map: Dict[str, str],
    ) -> PolicyResult:
        lang = lang_detection.primary

        tool_intent = self._detect_tool(message)
        if tool_intent:
            tool_result = self.tool_executor.execute(tool_intent.name, tool_intent.args, token_map)
            summary = tool_result.get("summary")
            return PolicyResult(
                lang=lang,
                route="tool",
                answer=summary,
                citations=[],
                tool_result=tool_result,
                confidence=1.0,
                top_doc_id=None,
            )

        search_response: SearchResponse = self.retriever.search(
            query=message,
            lang_hint=lang,
            top_k=3,
            use_sentence_transformers=use_sentence_transformers,
        )

        top_score = search_response.top_score
        citations = format_citations(search_response.results)

        if top_score < confidence_threshold:
            refusal = "I'm not confident enough to answer that accurately. I can escalate this to a human specialist if you'd like."
            return PolicyResult(
                lang=lang,
                route="escalate",
                answer=refusal,
                citations=[],
                tool_result=None,
                confidence=top_score,
                top_doc_id=search_response.top_item.id if search_response.top_item else None,
            )

        if confidence_threshold <= top_score < confidence_threshold + 0.05 and search_response.top_item:
            top_tag = search_response.top_item.tags[0] if search_response.top_item.tags else "the topic"
            clarifying = (
                f"I found information about {top_tag}, but I'd like a bit more detail. Could you confirm what exactly you want to know?"
                "\nExamples: "
                "'Please confirm my latest statement' or 'How do I change my UPI limit?'"
            )
            return PolicyResult(
                lang=lang,
                route="ask",
                answer=clarifying,
                citations=citations,
                tool_result=None,
                confidence=top_score,
                top_doc_id=search_response.top_item.id,
            )

        if not search_response.results:
            refusal = "I'm not confident enough to answer that accurately. I can escalate this to a human specialist if you'd like."
            return PolicyResult(
                lang=lang,
                route="escalate",
                answer=refusal,
                citations=[],
                tool_result=None,
                confidence=top_score,
                top_doc_id=None,
            )

        answer_body = search_response.top_item.content
        final_answer = render_answer(answer_body, search_response.results[:3])
        return PolicyResult(
            lang=lang,
            route="answer",
            answer=final_answer,
            citations=citations,
            tool_result=None,
            confidence=top_score,
            top_doc_id=search_response.top_item.id,
        )


class ToolExecutor:
    """Thin wrapper around the tool module to support detokenization."""

    def __init__(self, tools_module: Any) -> None:
        self.tools_module = tools_module

    def execute(self, name: str, args: Dict[str, Any], token_map: Dict[str, str]) -> Dict[str, Any]:
        if name == "block_card" and args.get("tokenized_card") == "":
            # Attempt to locate token in the message map
            token_candidates = [token for token in token_map if token.startswith("__CARD_")]
            if token_candidates:
                args = {**args, "tokenized_card": token_candidates[0]}

        tool = getattr(self.tools_module, name)
        result = tool(**args, token_map=token_map)
        return result
