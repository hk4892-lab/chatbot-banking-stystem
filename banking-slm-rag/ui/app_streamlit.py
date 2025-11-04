"""Streamlit operator console for the banking chatbot."""

from __future__ import annotations

import time
from typing import Any, Dict

import httpx
import streamlit as st


st.set_page_config(page_title='Banking SLM RAG Chatbot', layout='wide')


def _call_api(payload: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        response = client.post('/chat/turn', json=payload)
        response.raise_for_status()
        return response.json()


def init_state() -> None:
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def main() -> None:
    st.title('🏦 Banking Support Chatbot (SLM + RAG)')
    st.caption('Multilingual, PII-safe assistant for informational banking FAQs.')

    init_state()

    with st.sidebar:
        st.header('Settings')
        base_url = st.text_input('API Base URL', value='http://localhost:8000')
        k = st.slider('Top-K context', min_value=1, max_value=10, value=5)
        temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        lang = st.selectbox('Language', options=['AUTO', 'EN', 'HI', 'TA'], index=0)
        if st.button('Clear Chat'):
            st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    user_input = st.chat_input('Ask about banking services...')
    if user_input:
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.markdown(user_input)

        payload = {'text': user_input, 'k': k, 'lang': lang, 'temperature': temperature}
        with st.chat_message('assistant'):
            placeholder = st.empty()
            reply = ''
            try:
                start = time.perf_counter()
                response = _call_api(payload, base_url)
                latency = time.perf_counter() - start
                reply = response['reply']
                citations = response.get('citations', [])
                placeholder.success(reply)
                st.caption(f"Latency: {latency*1000:.0f} ms · Route: {response['route']} · Lang: {response['lang']}")
                if citations:
                    st.markdown('**Citations**')
                    for citation in citations:
                        st.markdown(f"- {citation['title']} ({citation['doc_id']})")
            except httpx.HTTPError as exc:
                placeholder.error(f'API error: {exc}')
                reply = 'API error'
        st.session_state.chat_history.append({'role': 'assistant', 'content': reply})


if __name__ == '__main__':  # pragma: no cover
    main()
