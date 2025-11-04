## Index Toolkit

Utilities to build and inspect the FAISS knowledge base index.

- `build_index.py`: create `index/kb.faiss` + `index/kb.meta.pkl` from a JSONL knowledge base.
- `retrieve.py`: lightweight CLI to inspect retrieval results for a query.

### Build

```bash
python index/build_index.py data/kb_chunks.sample.jsonl index/kb.faiss index/kb.meta.pkl
```

### Inspect

```bash
python index/retrieve.py "kitna upi limit hai" --k 3
```

Both scripts rely on environment variables defined in `.env` (see project root) and will fall back to defaults when unspecified.
