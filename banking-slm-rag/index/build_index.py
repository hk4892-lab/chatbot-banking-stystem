"""Build a FAISS index from knowledge base JSONL."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_chunks(path: Path) -> list[dict]:
    records = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def build(args: argparse.Namespace) -> None:
    kb_path = Path(args.kb_path)
    index_path = Path(args.index_path)
    meta_path = Path(args.meta_path)

    print(f'Loading knowledge base from {kb_path}...')
    chunks = load_chunks(kb_path)
    if not chunks:
        raise RuntimeError('Knowledge base is empty.')

    print(f'Loading embedding model {args.embedding_model}...')
    model = SentenceTransformer(args.embedding_model, device='cpu')
    texts = [entry['content'] for entry in chunks]
    vectors = model.encode(texts, convert_to_numpy=True)
    vectors = normalize_vectors(vectors.astype(np.float32))
    dimension = vectors.shape[1]

    print('Building FAISS index...')
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving index to {index_path}')
    faiss.write_index(index, str(index_path))

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Serializing metadata to {meta_path}')
    with meta_path.open('wb') as handle:
        pickle.dump(chunks, handle)

    print('Completed.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build FAISS index for the banking chatbot.')
    parser.add_argument('kb_path', help='Path to JSONL knowledge base.')
    parser.add_argument('index_path', help='Output path for FAISS index file.')
    parser.add_argument('meta_path', help='Output path for metadata pickle.')
    parser.add_argument('--embedding-model', default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    return parser.parse_args()


if __name__ == '__main__':  # pragma: no cover
    build(parse_args())
