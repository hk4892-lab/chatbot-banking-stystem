"""Streamlit UI for Banking Chatbot."""
import sys
from pathlib import Path
import requests
import streamlit as st

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="Banking Support Chatbot",
    page_icon="🏦",
    layout="wide",
)

# API endpoint
API_BASE = "http://localhost:8000"


def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def get_config():
    """Get configuration from API."""
    try:
        response = requests.get(f"{API_BASE}/config", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def send_message(text: str, k: int, lang: str, temperature: float):
    """Send message to chatbot API."""
    try:
        response = requests.post(
            f"{API_BASE}/chat/turn",
            json={
                "text": text,
                "k": k,
                "lang": lang,
                "temperature": temperature,
            },
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header
st.title("🏦 Banking Support Chatbot")
st.caption("Production-grade SLM + RAG system | Multilingual: EN/HI/TA")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check API status
    api_status = check_api_health()
    if api_status:
        st.success("✅ API Connected")
        config = get_config()
        if config:
            with st.expander("Model Info", expanded=False):
                st.text(f"SLM: {config['slm_model'].split('/')[-1]}")
                st.text(f"Embedding: {config['embedding_model'].split('/')[-1]}")
    else:
        st.error("❌ API Offline")
        st.info("Start API with: `make run-api`")
    
    st.divider()
    
    # Controls
    lang = st.selectbox(
        "Language",
        ["AUTO", "EN", "HI", "TA"],
        help="AUTO = automatic detection",
    )
    
    k = st.slider(
        "Top-K Documents",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of knowledge base chunks to retrieve",
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Generation randomness (lower = more focused)",
    )
    
    st.divider()
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Examples
    st.subheader("📝 Example Queries")
    examples = [
        "How to reset ATM PIN?",
        "What is UPI transaction limit?",
        "एटीएम पिन कैसे रीसेट करें?",
        "Fixed deposit rates sollunga",
    ]
    for example in examples:
        if st.button(example, key=f"ex_{example}", use_container_width=True):
            st.session_state.user_input = example

# Chat interface
chat_container = st.container()

# Display chat history
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show metadata for assistant messages
            if msg["role"] == "assistant" and "metadata" in msg:
                meta = msg["metadata"]
                
                cols = st.columns([1, 1, 1, 1])
                with cols[0]:
                    route_emoji = "✅" if meta["route"] == "answer" else "⚠️"
                    st.caption(f"{route_emoji} {meta['route'].title()}")
                with cols[1]:
                    st.caption(f"🌐 {meta['lang']}")
                with cols[2]:
                    st.caption(f"⏱️ {meta['latency_ms']}ms")
                with cols[3]:
                    pii_emoji = "🔒" if meta["safety"]["pii_masked"] else "🔓"
                    st.caption(f"{pii_emoji} PII")
                
                # Citations
                if meta.get("citations"):
                    with st.expander("📚 Sources", expanded=False):
                        for citation in meta["citations"]:
                            st.markdown(f"- **{citation['title']}** (`{citation['doc_id']}`)")

# Input area
user_input = st.chat_input("Ask about banking services...", key="chat_input")

# Handle example button clicks
if "user_input" in st.session_state:
    user_input = st.session_state.user_input
    del st.session_state.user_input

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)
    
    # Get response
    if not api_status:
        error_msg = "⚠️ API is offline. Please start the API server first."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        with chat_container:
            with st.chat_message("assistant"):
                st.error(error_msg)
    else:
        with st.spinner("🤔 Thinking..."):
            response = send_message(user_input, k, lang, temperature)
        
        if "error" in response:
            error_msg = f"❌ {response['error']}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with chat_container:
                with st.chat_message("assistant"):
                    st.error(error_msg)
        else:
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["reply"],
                "metadata": response,
            })
            
            # Display assistant message
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response["reply"])
                    
                    # Metadata
                    meta = response
                    cols = st.columns([1, 1, 1, 1])
                    with cols[0]:
                        route_emoji = "✅" if meta["route"] == "answer" else "⚠️"
                        st.caption(f"{route_emoji} {meta['route'].title()}")
                    with cols[1]:
                        st.caption(f"🌐 {meta['lang']}")
                    with cols[2]:
                        st.caption(f"⏱️ {meta['latency_ms']}ms")
                    with cols[3]:
                        pii_emoji = "🔒" if meta["safety"]["pii_masked"] else "🔓"
                        st.caption(f"{pii_emoji} PII")
                    
                    # Citations
                    if meta.get("citations"):
                        with st.expander("📚 Sources", expanded=False):
                            for citation in meta["citations"]:
                                st.markdown(f"- **{citation['title']}** (`{citation['doc_id']}`)")
    
    st.rerun()

# Footer
st.divider()
st.caption("🔒 Secure Banking Support | PII Auto-Redacted | No Live Transactions")
