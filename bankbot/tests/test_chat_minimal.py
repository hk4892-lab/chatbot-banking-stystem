from fastapi.testclient import TestClient

from bankbot.app.api import app


client = TestClient(app)


def test_chat_returns_reply():
    payload = {
        "messages": [
            {"role": "user", "content": "What is the minimum balance for a regular savings account?"}
        ]
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["reply"].strip() != ""
    assert data["source"] in {"rag", "slm", "tool", "clarify", "escalate"}
