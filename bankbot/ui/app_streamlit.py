"""Streamlit UI for the multilingual banking support chatbot."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"


def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    if "config" not in st.session_state:
        st.session_state.config = fetch_config()


def fetch_config() -> dict[str, Any]:
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{API_BASE}/config")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {
            "threshold": 0.18,
            "languages": ["AUTO", "EN", "HI", "TA"],
            "sentence_transformers_available": False,
            "kb_entries": 0,
        }


def send_to_backend(message: str, *, threshold: float, lang: str, use_st: bool, escalate: bool = False) -> Optional[dict[str, Any]]:
    payload = {
        "message": message,
        "session_id": st.session_state.session_id,
        "threshold": threshold,
        "lang": lang,
        "use_sentence_transformers": use_st,
        "escalate": escalate,
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{API_BASE}/chat/turn", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"Server returned an error: {exc.response.text}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to reach backend: {exc}")
    return None


def append_message(role: str, content: str, meta: dict[str, Any] | None = None) -> None:
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "meta": meta or {},
        }
    )


def render_messages() -> None:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        meta = message.get("meta", {})
        with st.chat_message("assistant" if role == "assistant" else "user"):
            st.markdown(content)
            if role == "assistant":
                route = meta.get("route", "-")
                confidence = meta.get("confidence")
                caption = f"Route: {route}"
                if confidence is not None:
                    caption += f" ? Confidence: {confidence:.2f}"
                st.caption(caption)
                citations = meta.get("citations") or []
                if citations:
                    st.markdown(
                        "**Citations:** "
                        + ", ".join(f"{item['title']} ({item['id']})" for item in citations)
                    )
                tool_result = meta.get("tool_result")
                if tool_result:
                    st.json(tool_result, expanded=False)


def process_user_message(message: str, *, threshold: float, lang: str, use_st: bool) -> None:
    append_message("user", message)
    response = send_to_backend(message, threshold=threshold, lang=lang, use_st=use_st)
    if response:
        append_message(
            "assistant",
            response.get("answer") or "(no response)",
            meta={
                "route": response.get("route", "-"),
                "confidence": response.get("confidence"),
                "citations": response.get("citations"),
                "tool_result": response.get("tool_result"),
            },
        )


def process_escalation(*, threshold: float, lang: str, use_st: bool) -> None:
    append_message("user", "Escalate to human support")
    response = send_to_backend(
        "Escalation requested", threshold=threshold, lang=lang, use_st=use_st, escalate=True
    )
    if response:
        append_message(
            "assistant",
            response.get("answer") or "Escalation acknowledged.",
            meta={
                "route": response.get("route", "escalate"),
                "confidence": response.get("confidence"),
                "citations": response.get("citations"),
                "tool_result": response.get("tool_result"),
            },
        )


def main() -> None:
    st.set_page_config(page_title="BankBot", layout="wide", page_icon="??")
    _init_state()

    config = st.session_state.config

    st.sidebar.title("BankBot Controls")
    selected_lang = st.sidebar.selectbox(
        "Language preference",
        options=config.get("languages", ["AUTO", "EN", "HI", "TA"]),
        index=0,
    )
    threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.5,
        value=float(config.get("threshold", 0.18)),
        step=0.01,
    )
    use_st = st.sidebar.checkbox(
        "Use sentence-transformers (if installed)",
        value=False,
        disabled=not config.get("sentence_transformers_available", False),
    )

    if st.sidebar.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid4())
        st.experimental_rerun()

    st.sidebar.markdown("---")
    st.sidebar.metric("Knowledge base entries", config.get("kb_entries", 0))

    st.title("Multilingual Banking Support Chatbot")
    st.info("This demo avoids storing your personal data. Don't share sensitive info.")

    c1, c2, c3 = st.columns(3)
    if c1.button("EMI calculator"):
        process_user_message("calculate emi P=500000 r=10 n=60", threshold=threshold, lang=selected_lang, use_st=use_st)
    if c2.button("Block card"):
        process_user_message("block my card immediately", threshold=threshold, lang=selected_lang, use_st=use_st)
    if c3.button("Check UPI limit"):
        process_user_message("UPI limit kitna hai?", threshold=threshold, lang=selected_lang, use_st=use_st)

    render_messages()

    if st.button("Escalate to human"):
        process_escalation(threshold=threshold, lang=selected_lang, use_st=use_st)

    user_input = st.chat_input("Send a message")
    if user_input:
        process_user_message(user_input, threshold=threshold, lang=selected_lang, use_st=use_st)


if __name__ == "__main__":
    main()
