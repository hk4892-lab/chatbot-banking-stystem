"""Tests for the policy router."""
from __future__ import annotations

from pathlib import Path

from app.core.language import LanguageDetection
from app.core.policy import PolicyRouter, ToolExecutor
from app.core.retrieval import Retriever
from app.tools import tools as tool_impl


def make_router() -> PolicyRouter:
    data_dir = Path(__file__).resolve().parents[1] / "app" / "data"
    retriever = Retriever(data_dir)
    executor = ToolExecutor(tool_impl)
    return PolicyRouter(retriever, executor)


def test_policy_escalates_low_confidence() -> None:
    router = make_router()
    detection = LanguageDetection(primary="EN", detected_scripts=["EN"], is_mixed=False)

    result = router.handle(
        message="Tell me about planet travel insurance",
        lang_detection=detection,
        confidence_threshold=0.3,
        use_sentence_transformers=False,
        token_map={},
    )

    assert result.route == "escalate"
    assert result.answer and "human" in result.answer.lower()


def test_policy_triggers_tool_for_emi() -> None:
    router = make_router()
    detection = LanguageDetection(primary="EN", detected_scripts=["EN"], is_mixed=False)

    result = router.handle(
        message="calculate emi P=500000 r=10 n=60",
        lang_detection=detection,
        confidence_threshold=0.18,
        use_sentence_transformers=False,
        token_map={},
    )

    assert result.route == "tool"
    assert result.tool_result is not None
    assert result.tool_result["tool"] == "calculate_emi"
    assert "emi" in result.tool_result
