import os

os.environ.setdefault("USE_SLM", "0")

from fastapi.testclient import TestClient

from bankbot.app.api import app


client = TestClient(app)


def test_chat_smoke_returns_reply():
    payload = {
        "messages": [
            {"role": "user", "content": "How can I check my account balance?"}
        ]
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["reply"].strip()
    assert data["source"] in {"tool", "rag", "slm_rag", "clarify", "escalate"}
