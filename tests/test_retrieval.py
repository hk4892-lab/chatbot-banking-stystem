"""Tests for FAISS retrieval."""
import pytest
import tempfile
import json
from pathlib import Path

from app.retrieval import build_index, FAISSRetriever


@pytest.fixture
def temp_kb_data():
    """Create temporary KB data for testing."""
    # Create temp KB chunks
    chunks = [
        {
            "doc_id": "test_001",
            "title": "UPI Limits",
            "content": "UPI transaction limit is 1 lakh per transaction.",
            "lang": "EN",
            "tags": ["upi"],
            "updated_on": "2025-01-01",
        },
        {
            "doc_id": "test_002",
            "title": "ATM PIN Reset",
            "content": "To reset ATM PIN visit any ATM and select Forgot PIN option.",
            "lang": "EN",
            "tags": ["atm", "pin"],
            "updated_on": "2025-01-01",
        },
        {
            "doc_id": "test_hi_001",
            "title": "यूपीआई लिमिट",
            "content": "यूपीआई लेनदेन सीमा प्रति लेनदेन 1 लाख रुपये है।",
            "lang": "HI",
            "tags": ["upi"],
            "updated_on": "2025-01-01",
        },
    ]
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_index(temp_kb_data):
    """Build temporary FAISS index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test.faiss"
        meta_path = Path(tmpdir) / "test.pkl"
        
        # Build index
        build_index(
            kb_chunks_path=temp_kb_data,
            output_index_path=str(index_path),
            output_meta_path=str(meta_path),
            embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )
        
        yield str(index_path), str(meta_path)


def test_build_index(temp_kb_data):
    """Test index building."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test.faiss"
        meta_path = Path(tmpdir) / "test.pkl"
        
        build_index(
            kb_chunks_path=temp_kb_data,
            output_index_path=str(index_path),
            output_meta_path=str(meta_path),
            embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )
        
        assert index_path.exists()
        assert meta_path.exists()


def test_retrieval_english(temp_index):
    """Test retrieval with English query."""
    index_path, meta_path = temp_index
    
    retriever = FAISSRetriever(
        index_path=index_path,
        meta_path=meta_path,
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    
    results = retriever.search("What is UPI transaction limit?", k=3)
    
    assert len(results) > 0
    assert results[0]["score"] > 0.2  # Should have decent similarity
    assert "UPI" in results[0]["title"] or "upi" in results[0]["content"].lower()


def test_retrieval_hindi(temp_index):
    """Test retrieval with Hindi query."""
    index_path, meta_path = temp_index
    
    retriever = FAISSRetriever(
        index_path=index_path,
        meta_path=meta_path,
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    
    results = retriever.search("यूपीआई की सीमा क्या है?", k=3)
    
    assert len(results) > 0
    assert results[0]["score"] > 0.2


def test_retrieval_hinglish(temp_index):
    """Test retrieval with Hinglish query."""
    index_path, meta_path = temp_index
    
    retriever = FAISSRetriever(
        index_path=index_path,
        meta_path=meta_path,
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    
    results = retriever.search("UPI ki limit kitni hai?", k=3)
    
    assert len(results) > 0
    assert results[0]["score"] > 0.2


def test_retrieval_top_k(temp_index):
    """Test retrieval respects k parameter."""
    index_path, meta_path = temp_index
    
    retriever = FAISSRetriever(
        index_path=index_path,
        meta_path=meta_path,
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    
    results = retriever.search("banking", k=2)
    
    assert len(results) <= 2
