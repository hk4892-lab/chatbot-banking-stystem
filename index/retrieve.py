#!/usr/bin/env python3
"""CLI tool to test retrieval."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval import FAISSRetriever
from app.config import settings


def main():
    """Main entry point for retrieval testing."""
    if len(sys.argv) < 2:
        print("Usage: python retrieve.py <query> [k]")
        print("Example: python retrieve.py 'How to reset ATM PIN?' 5")
        sys.exit(1)
    
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"Query: {query}")
    print(f"Retrieving top {k} results...\n")
    
    # Initialize retriever
    retriever = FAISSRetriever(
        index_path=settings.faiss_index_path,
        meta_path=settings.faiss_meta_path,
        embedding_model=settings.embedding_model,
    )
    
    # Search
    results = retriever.search(query, k=k)
    
    if not results:
        print("❌ No results found. Make sure index is built.")
        sys.exit(1)
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Score: {result['score']:.4f}")
        print(f"Doc ID: {result['doc_id']}")
        print(f"Title: {result['title']}")
        print(f"Language: {result['lang']}")
        print(f"Content: {result['content'][:200]}...")
        print()


if __name__ == "__main__":
    main()
