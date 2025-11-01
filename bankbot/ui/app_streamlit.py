"""Streamlit UI for the BankBot demo."""
from __future__ import annotations

from typing import Any, Dict, List

import requests
import streamlit as st


API_BASE = st.secrets.get("bankbot_api", "http://localhost:8000")

st.set_page_config(
    page_title="BankBot Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.header("Chat Settings")
language_option = st.sidebar.selectbox("Preferred language", options=["AUTO", "EN", "HI", "TA"], index=0)
confidence_threshold = st.sidebar.slider("Confidence threshold", min_value=0.05, max_value=0.6, value=0.18, step=0.01)
use_sentence_transformers = st.sidebar.checkbox(
    "Use sentence-transformers if available", value=False, help="Requires optional dependency"
)

st.sidebar.markdown("---")
st.sidebar.caption("Logs are stored locally only for this session. Clear the terminal to reset.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def call_backend(message: str) -> Dict[str, Any]:
    payload = {
        "message": message,
        "lang": language_option,
        "confidence_threshold": confidence_threshold,
        "use_sentence_transformers": use_sentence_transformers,
    }
    response = requests.post(f"{API_BASE}/chat/turn", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


st.markdown(
    "<div style='border:1px solid #1db954;padding:0.75rem;border-radius:0.5rem;background:#16202f;'>"
    "<strong>Safety:</strong> This demo avoids storing your personal data. Please do not share sensitive information."  # noqa: E501
    "</div>",
    unsafe_allow_html=True,
)

quick_cols = st.columns(3)
quick_actions = {
    "EMI calculator": "calculate emi P=500000 r=10 n=60",
    "Block card": "block my card because it was stolen",
    "Check UPI limit": "UPI limit kitna hai",
}

for col, (label, template) in zip(quick_cols, quick_actions.items()):
    if col.button(label):
        st.session_state.quick_prompt = template


if "quick_prompt" in st.session_state:
    default_input = st.session_state.pop("quick_prompt")
else:
    default_input = ""

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            meta = message.get("meta", {})
            st.caption(
                f"Route: {meta.get('route', 'answer')} | Confidence: {meta.get('confidence', 0):.2f}"
            )
            citations = meta.get("citations", [])
            if citations:
                citation_text = ", ".join(f"{c['title']} ({c['id']})" for c in citations)
                st.caption(f"Sources: {citation_text}")
            tool_result = meta.get("tool_result")
            if tool_result:
                with st.expander("Tool result"):
                    st.json(tool_result)


prompt = st.chat_input("Ask about your banking needs", value=default_input)

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        backend_response = call_backend(prompt)
    except requests.HTTPError as exc:
        error_message = f"Backend error: {exc.response.status_code}"
        st.session_state["messages"].append({"role": "assistant", "content": error_message})
        with st.chat_message("assistant"):
            st.error(error_message)
    except requests.RequestException as exc:
        error_message = f"Connection issue: {exc}"
        st.session_state["messages"].append({"role": "assistant", "content": error_message})
        with st.chat_message("assistant"):
            st.error(error_message)
    else:
        answer = backend_response.get("answer") or "(No answer provided.)"
        meta = {
            "route": backend_response.get("route"),
            "confidence": backend_response.get("confidence", 0.0),
            "citations": backend_response.get("citations", []),
            "tool_result": backend_response.get("tool_result"),
        }
        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "meta": meta}
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(f"Route: {meta['route']} | Confidence: {meta['confidence']:.2f}")
            if meta["citations"]:
                citation_text = ", ".join(f"{c['title']} ({c['id']})" for c in meta["citations"])
                st.caption(f"Sources: {citation_text}")
            if meta["tool_result"]:
                with st.expander("Tool result"):
                    st.json(meta["tool_result"])

st.markdown("---")
if st.button("Escalate to human"):
    st.info("A human specialist will reach out shortly (demo placeholder).")
