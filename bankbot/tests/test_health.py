import os

os.environ.setdefault("USE_SLM", "0")

from fastapi.testclient import TestClient

from bankbot.app.api import app


client = TestClient(app)


def test_health_endpoint_returns_status():
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert isinstance(payload["rag"], bool)
    assert isinstance(payload["slm"], bool)
    assert isinstance(payload["kb_size"], int)
    assert isinstance(payload["langs"], list)
