# Banking Support Chatbot - SLM + RAG

A **production-grade** multilingual banking support chatbot using Small Language Models (SLM) and Retrieval-Augmented Generation (RAG). Built for informational banking FAQs with strong PII protection, no live transactions, and support for English, Hindi, and Tamil.

## 🎯 Features

- **RAG Architecture**: FAISS vector search + SentenceTransformers for semantic retrieval
- **SLM Generation**: Microsoft Phi-3 or Qwen2.5 for grounded, citation-backed responses
- **Multilingual**: English, Hindi, Tamil with code-mix support (Hinglish, Tamlish)
- **PII Protection**: Deterministic redaction for phone, email, PAN, Aadhaar, card numbers
- **Safety-First**: Refuses balance/transaction queries; informational guidance only
- **Production-Ready**: FastAPI + Uvicorn, Docker, Prometheus metrics, audit logging
- **CPU-Friendly**: Works on CPU; auto-detects CUDA if available

## 🏗️ Architecture

```
┌─────────────┐
│   User      │
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  1. PII Redaction (phone, email, PAN, etc.)    │
└──────────────────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  2. Language Detection & Normalization          │
│     (EN/HI/TA, Hinglish/Tamlish synonyms)      │
└──────────────────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  3. Retrieval (FAISS + Embeddings)             │
│     - Semantic search over KB chunks            │
│     - Top-k most relevant documents             │
└──────────────────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  4. Route Decision                              │
│     - Sufficient evidence? → Answer             │
│     - Sensitive/forbidden? → Refusal            │
└──────────────────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  5. Grounded Prompt + SLM Generation           │
│     - System: Answer from CONTEXT only          │
│     - Context: Top-k retrieved chunks           │
│     - User: Masked query                        │
└──────────────────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  6. Response with Citations [Title (doc_id)]   │
└─────────────────────────────────────────────────┘
```

## 🚀 Quickstart

### 1. Setup

```bash
# Clone and enter directory
cd /workspace

# Create virtual environment and install dependencies
make venv
source venv/bin/activate

# Copy environment config
cp .env.example .env
```

### 2. Build FAISS Index

```bash
# Build index from sample KB chunks
make index

# This creates:
#   index/kb.faiss - FAISS vector index
#   index/kb.meta.pkl - Document metadata
```

### 3. Start API Server

```bash
# Start FastAPI server (dev mode with auto-reload)
make run-api

# Server runs at: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### 4. Launch UI (Optional)

```bash
# In a new terminal (with venv activated)
make ui

# Streamlit UI at: http://localhost:8501
```

### 5. Test

```bash
# Run test suite
make test

# Lint code
make lint
```

## 📁 Project Structure

```
banking-slm-rag/
├─ app/                        # Core application
│  ├─ api.py                   # FastAPI endpoints
│  ├─ config.py                # Settings management
│  ├─ deps.py                  # Dependency injection
│  ├─ schemas.py               # Pydantic models
│  ├─ redaction.py             # PII masking
│  ├─ language.py              # Language detection/normalization
│  ├─ retrieval.py             # FAISS retrieval
│  ├─ generator.py             # SLM generation
│  └─ pipeline.py              # End-to-end RAG pipeline
├─ ui/
│  └─ app_streamlit.py         # Streamlit UI
├─ data/
│  ├─ kb_chunks.sample.jsonl   # Sample knowledge base (30 chunks)
│  ├─ sft_train.sample.jsonl   # SFT training data
│  └─ sft_val.sample.jsonl     # SFT validation data
├─ index/
│  ├─ build_index.py           # Build FAISS index script
│  ├─ retrieve.py              # CLI retrieval tester
│  └─ README.md
├─ train/
│  ├─ sft_lora.py              # LoRA fine-tuning script
│  └─ README.md
├─ eval/
│  ├─ evaluate.py              # Evaluation script
│  ├─ prompts.sample.csv       # Eval queries
│  └─ metrics.md
├─ tests/                      # Test suite
│  ├─ test_redaction.py
│  ├─ test_retrieval.py
│  ├─ test_pipeline.py
│  ├─ test_api.py
│  └─ test_language.py
├─ .env.example                # Environment config template
├─ requirements.txt            # Python dependencies
├─ Makefile                    # Common commands
├─ Dockerfile                  # Container image
└─ README.md                   # This file
```

## 🔧 Configuration

Edit `.env` or set environment variables:

```bash
# Models
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
SLM_MODEL=microsoft/phi-3-mini-4k-instruct

# Paths
FAISS_INDEX_PATH=index/kb.faiss
FAISS_META_PATH=index/kb.meta.pkl
KB_CHUNKS_PATH=data/kb_chunks.sample.jsonl

# Generation
DEFAULT_K=5
DEFAULT_TEMPERATURE=0.3
DEFAULT_TOP_P=0.9
DEFAULT_MAX_TOKENS=256

# Safety
REFUSAL_THRESHOLD=0.2
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW_SECONDS=60

# Server
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost,http://localhost:8501
```

## 📡 API Endpoints

### `POST /chat/turn`

Main chat endpoint.

**Request:**
```json
{
  "text": "How to reset ATM PIN?",
  "k": 5,
  "lang": "AUTO",
  "temperature": 0.3
}
```

**Response:**
```json
{
  "reply": "To reset your ATM PIN: 1) Visit any bank ATM...",
  "citations": [
    {"doc_id": "kb_en_002", "title": "ATM PIN Reset Procedure"}
  ],
  "used_context_ids": ["kb_en_002", "kb_en_005"],
  "route": "answer",
  "latency_ms": 1523,
  "lang": "EN",
  "safety": {
    "pii_masked": false,
    "refusal": false
  }
}
```

### `GET /health`

Health check: `{"status": "ok"}`

### `GET /config`

Get system configuration (models, defaults, languages).

### `GET /metrics`

Prometheus-compatible metrics:
- `chatbot_requests_total` - Request counts by endpoint and route
- `chatbot_latency_seconds` - Latency histogram
- `chatbot_refusals_total` - Refusal count

### `GET /debug/prompt`

Debug endpoint to inspect masked prompt (useful for development).

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_redaction.py -v

# Run with coverage
pytest --cov=app tests/
```

**Test Coverage:**
- ✅ PII redaction (phone, email, PAN, Aadhaar, cards)
- ✅ Language detection (EN/HI/TA)
- ✅ Retrieval (English, Hindi, Hinglish queries)
- ✅ Pipeline (answer/refusal routes, citations)
- ✅ API endpoints (validation, responses)

## 📊 Evaluation

Evaluate retrieval quality:

```bash
# Build index first
make index

# Run evaluation
python eval/evaluate.py eval/prompts.sample.csv 5
```

**Metrics:**
- **Recall@k**: Proportion of queries retrieving expected doc in top-k
- **nDCG@k**: Ranking quality (higher = better position)
- **Citation coverage**: % responses with proper citations

**Target Performance:**
- Recall@5 > 0.85
- nDCG@5 > 0.80

See `eval/metrics.md` for detailed metric definitions.

## 🎓 Fine-Tuning (Optional)

Fine-tune the SLM with LoRA for better banking-specific responses:

```bash
# Train LoRA adapter
make train-lora

# Or with custom data
python train/sft_lora.py data/sft_train.jsonl data/sft_val.jsonl out/my-lora
```

**Benefits:**
- Better instruction following
- Improved multilingual handling
- Consistent refusal behavior
- Proper citation format

See `train/README.md` for detailed guide.

## 🔒 Safety & PII Policy

### PII Redaction

All queries are automatically scanned and masked:

| Type | Pattern | Token |
|------|---------|-------|
| Phone | `9876543210` | `PII_PHONE_1` |
| Email | `user@example.com` | `PII_EMAIL_1` |
| PAN | `ABCDE1234F` | `PII_PAN_1` |
| Aadhaar | `1234 5678 9012` | `PII_AADHAAR_1` |
| Card | `1234-5678-9012-3456` | `PII_CARD_1` |
| Account | `123456789012` | `PII_ACCOUNT_1` |

**Guarantees:**
- Raw PII never sent to LLM
- Only masked tokens in logs
- Deterministic token assignment

### Forbidden Queries

System refuses queries containing:
- Balance checks
- Transaction history
- Money transfers
- OTP/PIN generation
- Personal account details

**Refusal Response:**
> "I apologize, but I don't have enough information to answer your question accurately. Please contact our customer service at 1800-XXX-XXXX or visit your nearest branch."

### Audit Logging

All interactions logged to `logs/audit.jsonl` (PII-scrubbed):

```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "event": "chat_turn",
  "route": "answer",
  "lang": "EN",
  "latency_ms": 1200,
  "pii_masked": true,
  "client_ip": "127.0.0.1"
}
```

## 🌐 Multilingual Support

### Supported Languages

| Code | Language | Script |
|------|----------|--------|
| EN | English | Latin |
| HI | Hindi | Devanagari |
| TA | Tamil | Tamil |

### Code-Mix Handling

**Hinglish** (Hindi + English):
- "UPI ki limit kitna hai?" → "UPI ki limit how much hai?"
- Auto-detection + synonym normalization

**Tamlish** (Tamil + English):
- "Card eppadi block pannunga?" → "Card how block please?"
- Synonym expansion before retrieval

### Adding More Languages

1. Add language chunks to `data/kb_chunks.sample.jsonl`
2. Update `app/language.py` with Unicode ranges and synonyms
3. Rebuild index: `make index`

## 📦 Docker Deployment

### Build Image

```bash
docker build -t banking-chatbot:latest .
```

### Run Container

```bash
# Create index directory (or mount existing)
mkdir -p index logs

# Run API
docker run -d \
  --name banking-chatbot \
  -p 8000:8000 \
  -v $(pwd)/index:/app/index \
  -v $(pwd)/logs:/app/logs \
  banking-chatbot:latest
```

### Docker Compose (recommended)

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./index:/app/index
      - ./logs:/app/logs
    environment:
      - SLM_MODEL=microsoft/phi-3-mini-4k-instruct
      - RATE_LIMIT_REQUESTS=100
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 🔍 Adding New Knowledge

### 1. Edit KB Chunks

Add to `data/kb_chunks.sample.jsonl`:

```json
{
  "doc_id": "kb_en_014",
  "title": "Your New Topic",
  "content": "Detailed information about banking topic...",
  "lang": "EN",
  "tags": ["tag1", "tag2"],
  "updated_on": "2025-01-15"
}
```

### 2. Rebuild Index

```bash
make index
```

### 3. Restart API

```bash
# If running with make
Ctrl+C, then make run-api

# If Docker
docker restart banking-chatbot
```

## 📈 Performance

### Benchmarks (Sample Hardware)

| Hardware | Latency (P50) | Latency (P95) | Throughput |
|----------|---------------|---------------|------------|
| CPU (8 cores) | 2.5s | 5.0s | ~10 QPS |
| GPU (RTX 3090) | 450ms | 800ms | ~50 QPS |

**Optimization Tips:**
- Use GPU for SLM (5-10x faster)
- Increase `DEFAULT_MAX_TOKENS` cautiously (impacts latency)
- Consider model quantization (int8, int4) for faster CPU inference
- Cache embeddings for common queries

## 🐛 Troubleshooting

### API won't start

```bash
# Check if index is built
ls -lh index/

# Rebuild if missing
make index

# Check dependencies
pip install -r requirements.txt
```

### Out of Memory

```bash
# Use smaller model
export SLM_MODEL=microsoft/phi-2

# Or quantize (requires bitsandbytes)
# Edit app/generator.py to add load_in_8bit=True
```

### Low retrieval accuracy

```bash
# Test retrieval directly
python index/retrieve.py "your test query" 10

# Check if KB has relevant content
cat data/kb_chunks.sample.jsonl | grep "keyword"

# Try different embedding model
export EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### Model downloads failing

```bash
# Pre-download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')"
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/phi-3-mini-4k-instruct')"
```

## 📚 References

- **FAISS**: [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- **SentenceTransformers**: [www.sbert.net](https://www.sbert.net)
- **Phi-3**: [huggingface.co/microsoft/phi-3-mini-4k-instruct](https://huggingface.co/microsoft/phi-3-mini-4k-instruct)
- **TRL**: [github.com/huggingface/trl](https://github.com/huggingface/trl)
- **PEFT**: [github.com/huggingface/peft](https://github.com/huggingface/peft)

## 🤝 Contributing

1. Add new KB chunks with sources
2. Expand test coverage
3. Improve multilingual normalization
4. Optimize prompts for better citations
5. Add more evaluation metrics

## 📄 License

MIT License - Feel free to use for your banking chatbot projects!

## 🎯 Roadmap

- [ ] Add Bengali, Marathi support
- [ ] Cross-encoder reranking for better retrieval
- [ ] Voice input/output
- [ ] Hybrid search (keyword + semantic)
- [ ] Model distillation for faster inference
- [ ] A/B testing framework

---

**Built with ❤️ for production banking support**

For questions or issues, see troubleshooting section above or check `train/README.md`, `index/README.md`, `eval/metrics.md` for specific topics.
