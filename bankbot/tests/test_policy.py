from __future__ import annotations

from pathlib import Path

from app.core.language import Language
from app.core.policy import ConversationState, DialoguePolicy
from app.core.retrieval import RetrievalEngine


DATA_DIR = Path(__file__).resolve().parents[1] / "app" / "data"


def make_policy() -> DialoguePolicy:
    retriever = RetrievalEngine(DATA_DIR, use_sentence_transformers=False)
    return DialoguePolicy(retriever)


def test_policy_invokes_emi_tool() -> None:
    policy = make_policy()
    session = ConversationState()

    decision = policy.decide("calculate emi P=500000 r=10 n=60", Language.EN, session)

    assert decision.route == "tool"
    assert decision.tool_result
    assert decision.tool_result["tool"] == "calculate_emi"


def test_policy_escalates_on_low_score() -> None:
    policy = make_policy()
    session = ConversationState()

    decision = policy.decide("Tell me about international travel insurance", Language.EN, session)

    assert decision.route in {"ask", "escalate"}
