"""Tests for FastAPI endpoints."""
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_pipeline():
    """Mock pipeline for API tests."""
    with patch("app.api.get_pipeline") as mock:
        pipeline = Mock()
        pipeline.answer = Mock(return_value={
            "reply": "UPI limit is ₹1,00,000.",
            "citations": [{"doc_id": "kb_001", "title": "UPI Limits"}],
            "used_context_ids": ["kb_001"],
            "route": "answer",
            "latency_ms": 150,
            "lang": "EN",
            "safety": {"pii_masked": False, "refusal": False},
        })
        mock.return_value = pipeline
        yield pipeline


def test_health_endpoint(client):
    """Test /health endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config_endpoint(client):
    """Test /config endpoint."""
    response = client.get("/config")
    
    assert response.status_code == 200
    data = response.json()
    assert "embedding_model" in data
    assert "slm_model" in data
    assert "default_k" in data
    assert "languages" in data


def test_chat_turn_success(client, mock_pipeline):
    """Test /chat/turn with successful response."""
    response = client.post(
        "/chat/turn",
        json={"text": "What is UPI limit?", "k": 5, "lang": "AUTO", "temperature": 0.3},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "reply" in data
    assert "citations" in data
    assert "route" in data
    assert "latency_ms" in data
    assert "lang" in data
    assert "safety" in data
    assert data["route"] in ["answer", "refusal"]
    assert data["latency_ms"] > 0


def test_chat_turn_with_citations(client, mock_pipeline):
    """Test /chat/turn returns citations."""
    response = client.post(
        "/chat/turn",
        json={"text": "What is UPI limit?"},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "citations" in data
    assert isinstance(data["citations"], list)
    if data["route"] == "answer":
        # Should have citations for answer route
        for citation in data["citations"]:
            assert "doc_id" in citation
            assert "title" in citation


def test_chat_turn_validation(client):
    """Test /chat/turn request validation."""
    # Empty text
    response = client.post("/chat/turn", json={"text": ""})
    assert response.status_code == 422
    
    # Invalid k
    response = client.post("/chat/turn", json={"text": "test", "k": 0})
    assert response.status_code == 422
    
    # Invalid lang
    response = client.post("/chat/turn", json={"text": "test", "lang": "FR"})
    assert response.status_code == 422
    
    # Invalid temperature
    response = client.post("/chat/turn", json={"text": "test", "temperature": 2.0})
    assert response.status_code == 422


def test_chat_turn_defaults(client, mock_pipeline):
    """Test /chat/turn uses defaults."""
    response = client.post("/chat/turn", json={"text": "test query"})
    
    assert response.status_code == 200
    # Should use default values for k, lang, temperature


def test_metrics_endpoint(client):
    """Test /metrics endpoint returns Prometheus format."""
    response = client.get("/metrics")
    
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    # Should contain metric names
    text = response.text
    assert "chatbot_requests_total" in text or "# HELP" in text


def test_debug_prompt_endpoint(client, mock_pipeline):
    """Test /debug/prompt endpoint."""
    mock_pipeline.retriever = Mock()
    mock_pipeline.retriever.search = Mock(return_value=[
        {
            "doc_id": "kb_001",
            "title": "Test",
            "content": "Test content",
            "lang": "EN",
            "score": 0.8,
        }
    ])
    mock_pipeline.generator = Mock()
    mock_pipeline.generator.make_prompt = Mock(return_value="<test prompt>")
    
    response = client.get("/debug/prompt?text=test&k=3")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "original" in data
    assert "masked" in data
    assert "lang" in data
    assert "prompt" in data
