# Index Building and Retrieval

This directory contains scripts for building the FAISS index and testing retrieval.

## Building the Index

The index must be built before running the chatbot. This creates a FAISS vector index from your knowledge base chunks.

```bash
# Using Makefile
make index

# Or directly
python index/build_index.py data/kb_chunks.sample.jsonl index/kb.faiss index/kb.meta.pkl
```

**What happens:**
1. Loads all KB chunks from JSONL file
2. Generates embeddings using SentenceTransformers
3. Creates FAISS IndexFlatIP (cosine similarity with normalized vectors)
4. Saves index and metadata

## Testing Retrieval

Test the search functionality with queries:

```bash
# Basic search
python index/retrieve.py "How to reset ATM PIN?" 5

# Hindi query
python index/retrieve.py "एटीएम पिन कैसे रीसेट करें?" 3

# Tamil query
python index/retrieve.py "எப்படி கார்டை பிளாக் செய்வது?" 5
```

## Adding New Knowledge

To add new banking FAQs:

1. **Edit KB chunks**: Add entries to `data/kb_chunks.sample.jsonl` following this format:

```json
{
  "doc_id": "kb_en_014",
  "title": "Your Topic Title",
  "content": "Detailed information about the topic...",
  "lang": "EN",
  "tags": ["tag1", "tag2"],
  "updated_on": "2025-01-15"
}
```

2. **Rebuild index**: Run `make index` to regenerate the FAISS index with new content.

3. **Restart API**: Restart the FastAPI server to load the new index.

## Index Format

- **FAISS Index** (`kb.faiss`): Binary file with vector embeddings
- **Metadata** (`kb.meta.pkl`): Pickle file with document metadata (doc_id, title, content, lang, tags, updated_on)

## Performance

- CPU-friendly: Works on machines without GPU
- Fast search: ~1-2ms for top-k retrieval on 30-50 documents
- Scalable: Can handle thousands of chunks (consider IVF indexes for 100k+ docs)
