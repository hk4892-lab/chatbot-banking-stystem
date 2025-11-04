"""Tests for RAG pipeline."""
import pytest
from unittest.mock import Mock, MagicMock

from app.pipeline import RAGPipeline


@pytest.fixture
def mock_retriever():
    """Mock FAISS retriever."""
    retriever = Mock()
    retriever.search = Mock(return_value=[
        {
            "doc_id": "kb_001",
            "title": "UPI Limits",
            "content": "UPI transaction limit is 1 lakh per transaction.",
            "lang": "EN",
            "score": 0.85,
        }
    ])
    return retriever


@pytest.fixture
def mock_generator():
    """Mock SLM generator."""
    generator = Mock()
    generator.make_prompt = Mock(return_value="<prompt>")
    generator.generate = Mock(return_value="UPI transaction limit is ₹1,00,000 per transaction. [UPI Limits (kb_001)]")
    return generator


@pytest.fixture
def pipeline(mock_retriever, mock_generator):
    """Create pipeline with mocked components."""
    return RAGPipeline(retriever=mock_retriever, generator=mock_generator)


def test_pipeline_answer_route(pipeline):
    """Test pipeline with relevant query returns answer."""
    result = pipeline.answer("What is UPI limit?", k=5, lang="AUTO", temperature=0.3)
    
    assert result["route"] == "answer"
    assert result["reply"]
    assert len(result["citations"]) > 0
    assert result["lang"] in ["EN", "HI", "TA"]
    assert result["latency_ms"] > 0
    assert isinstance(result["safety"]["pii_masked"], bool)
    assert isinstance(result["safety"]["refusal"], bool)


def test_pipeline_pii_masking(pipeline):
    """Test pipeline masks PII."""
    result = pipeline.answer("My phone is 9876543210, what is UPI limit?", k=5)
    
    assert result["safety"]["pii_masked"] is True


def test_pipeline_refusal_no_context(pipeline, mock_retriever):
    """Test pipeline refuses when no relevant context."""
    # Mock low-score results
    mock_retriever.search = Mock(return_value=[
        {
            "doc_id": "kb_001",
            "title": "Irrelevant",
            "content": "Some text",
            "lang": "EN",
            "score": 0.05,  # Below threshold
        }
    ])
    
    result = pipeline.answer("Random irrelevant query xyz", k=5)
    
    assert result["route"] == "refusal"
    assert result["safety"]["refusal"] is True


def test_pipeline_refusal_sensitive(pipeline):
    """Test pipeline refuses sensitive queries."""
    result = pipeline.answer("Show my account balance", k=5)
    
    # Should refuse due to sensitive keywords
    assert result["route"] == "refusal"
    assert result["safety"]["refusal"] is True


def test_pipeline_language_detection(pipeline):
    """Test pipeline detects language."""
    # English
    result = pipeline.answer("How to reset PIN?", k=5)
    assert result["lang"] == "EN"
    
    # Hindi
    result = pipeline.answer("पिन कैसे रीसेट करें?", k=5)
    assert result["lang"] == "HI"


def test_pipeline_citations(pipeline):
    """Test pipeline extracts citations."""
    result = pipeline.answer("What is UPI limit?", k=5)
    
    assert "citations" in result
    assert len(result["citations"]) > 0
    assert all("doc_id" in c and "title" in c for c in result["citations"])


def test_pipeline_context_ids(pipeline):
    """Test pipeline tracks used context."""
    result = pipeline.answer("What is UPI limit?", k=5)
    
    if result["route"] == "answer":
        assert len(result["used_context_ids"]) > 0
    else:
        assert len(result["used_context_ids"]) == 0
