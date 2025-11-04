#!/usr/bin/env python3
"""Build FAISS index from KB chunks."""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval import build_index
from app.config import settings


def main():
    """Main entry point for index building."""
    if len(sys.argv) < 4:
        print("Usage: python build_index.py <kb_chunks.jsonl> <output_index.faiss> <output_meta.pkl>")
        print(f"Example: python build_index.py data/kb_chunks.sample.jsonl index/kb.faiss index/kb.meta.pkl")
        sys.exit(1)
    
    kb_chunks_path = sys.argv[1]
    output_index_path = sys.argv[2]
    output_meta_path = sys.argv[3]
    
    print(f"Building FAISS index from: {kb_chunks_path}")
    print(f"Using embedding model: {settings.embedding_model}")
    print(f"Output index: {output_index_path}")
    print(f"Output metadata: {output_meta_path}")
    
    # Ensure output directory exists
    Path(output_index_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Build index
    build_index(
        kb_chunks_path=kb_chunks_path,
        output_index_path=output_index_path,
        output_meta_path=output_meta_path,
        embedding_model=settings.embedding_model,
    )
    
    print("\n✅ Index built successfully!")


if __name__ == "__main__":
    main()
