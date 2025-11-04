"""CLI helper to inspect retrieval results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.retrieval import FaissRetriever


def run(query: str, k: int) -> None:
    settings = get_settings()
    model = SentenceTransformer(settings.embedding_model, device='cpu')
    retriever = FaissRetriever(settings=settings, embedding_model=model)
    results = retriever.search(query, k)
    for idx, result in enumerate(results, start=1):
        payload = {
            'rank': idx,
            'score': round(result.score, 4),
            'doc_id': result.chunk.doc_id,
            'title': result.chunk.title,
            'content': result.chunk.content,
        }
        print(json.dumps(payload, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description='Query the FAISS index for debugging.')
    parser.add_argument('query', help='Search query text.')
    parser.add_argument('--k', type=int, default=5, help='Number of passages to return.')
    args = parser.parse_args()
    run(args.query, args.k)


if __name__ == '__main__':  # pragma: no cover
    main()
