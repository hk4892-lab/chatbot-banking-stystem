# System Architecture

## Overview

The Banking SLM-RAG Chatbot is a production-grade conversational AI system designed for banking customer support. It combines Retrieval-Augmented Generation (RAG) with Small Language Models (SLM) to provide accurate, grounded, and safe responses to banking queries.

## Design Principles

1. **Safety First**: All PII redacted before processing; no live transactions
2. **Grounded Responses**: Every answer backed by retrieved knowledge base
3. **Multilingual**: Native support for English, Hindi, Tamil with code-mix
4. **Deterministic**: No randomness in PII handling or routing decisions
5. **Observable**: Full metrics, logging, and audit trail
6. **CPU-Friendly**: Works without GPU (optimized for CPU by default)

## System Components

### 1. Query Processing Pipeline

```
User Query → PII Redaction → Language Detection → Normalization → Retrieval
```

**PII Redaction** (`app/redaction.py`)
- Scans for: phone, email, PAN, Aadhaar, card/account numbers
- Replaces with deterministic tokens (e.g., `PII_PHONE_1`)
- Maintains mapping for potential unmasking (never used in production)

**Language Detection** (`app/language.py`)
- Unicode range analysis (Devanagari → HI, Tamil → TA, else EN)
- Threshold: > 3 non-Latin characters triggers regional language
- Code-mix aware: Hinglish/Tamlish handled via synonym expansion

**Normalization**
- Whitespace collapse
- Synonym expansion (e.g., "kitna" → "how much")
- Case normalization (language-specific)

### 2. Retrieval System

```
Normalized Query → Embedding → FAISS Search → Top-K Documents
```

**Embedding Model**
- `paraphrase-multilingual-mpnet-base-v2` (768-dim vectors)
- Normalized embeddings for cosine similarity via inner product
- ~100ms encoding time on CPU

**FAISS Index**
- `IndexFlatIP`: Inner product for normalized vectors (= cosine similarity)
- Exact search (no approximation for small KB)
- Scales to 10k+ documents without IVF

**Metadata Storage**
- Pickle file with doc_id, title, content, lang, tags, updated_on
- Synchronized with FAISS index by position

### 3. Routing Decision

```
Retrieved Docs → Score Check → Route (Answer | Refusal)
```

**Routing Logic**:
1. Check average score of top-3 docs
2. If avg_score < `REFUSAL_THRESHOLD` (0.2) → Refusal
3. If `has_sensitive_keywords()` → Refusal
4. Else → Answer

**Sensitive Keywords**:
- balance, transaction, transfer, OTP, PIN, password
- Multilingual variants (बैलेंस, பணம், etc.)

### 4. Generation System

```
Context + Query → Prompt → SLM → Response → Citation Extraction
```

**Prompt Structure**:
```
<|system|>
You are a banking assistant. Answer strictly from CONTEXT.
Cite as [Title (doc_id)]. Never unmask PII.

CONTEXT:
[UPI Limits (kb_en_001)]
UPI transaction limit is ₹1,00,000...

<|user|>
What is UPI limit?

<|assistant|>
```

**SLM Configuration**:
- Model: Phi-3-mini-4k-instruct (3.8B parameters)
- Temperature: 0.3 (low for factual responses)
- Top-p: 0.9
- Max tokens: 256
- Device: CPU (float32) or CUDA (float16)

**Citation Extraction**:
- Regex pattern: `\[([^\]]+)\s+\(([^)]+)\)\]`
- Fallback: Use top-3 retrieved docs if no explicit citations

### 5. API Layer

**FastAPI Application** (`app/api.py`)
- CORS middleware (configurable origins)
- Rate limiting (in-memory token bucket)
- Prometheus metrics export
- Health checks

**Endpoints**:
- `POST /chat/turn`: Main chat interface
- `GET /health`: Health check
- `GET /config`: System configuration
- `GET /metrics`: Prometheus metrics
- `GET /debug/prompt`: Debug masked prompt

**Metrics Tracked**:
- `chatbot_requests_total`: Counter by endpoint and route
- `chatbot_latency_seconds`: Histogram with buckets
- `chatbot_refusals_total`: Counter for safety tracking

### 6. Observability

**Audit Logging** (`logs/audit.jsonl`)
- One JSON per line
- PII-scrubbed before writing
- Fields: timestamp, event, route, lang, latency_ms, pii_masked, client_ip

**Prometheus Metrics** (`/metrics`)
- Request counts (by route, endpoint)
- Latency percentiles (P50, P95, P99)
- Refusal rate
- Standard format for Grafana/Prometheus integration

## Data Flow

### Successful Query

```
1. User: "How to reset ATM PIN?"
2. Redaction: No PII detected
3. Language: Detected as EN
4. Normalization: "how to reset atm pin?"
5. Retrieval: Top-5 docs retrieved, avg_score = 0.87
6. Route: ANSWER (score >= 0.2)
7. Prompt: System + Context(5 docs) + User query
8. Generation: "To reset ATM PIN: 1) Visit ATM... [ATM PIN Reset (kb_en_002)]"
9. Response: {reply, citations, route="answer", ...}
```

### Query with PII

```
1. User: "My phone 9876543210, reset PIN?"
2. Redaction: "My phone PII_PHONE_1, reset PIN?"
3. Language: EN
4. Retrieval: Uses masked query
5. Generation: Uses masked query in prompt
6. Response: Contains no raw phone number
```

### Forbidden Query

```
1. User: "Show my account balance"
2. Redaction: No PII
3. Language: EN
4. Retrieval: Low relevance docs retrieved
5. has_sensitive_keywords: TRUE ("balance" detected)
6. Route: REFUSAL
7. Response: Localized refusal message
```

## Security Guarantees

### PII Protection

- **Never logged raw**: All logging goes through `scrub_log()`
- **Never sent to LLM**: Redaction before prompt building
- **Deterministic masking**: Same PII → same token in session
- **No unmasking**: System has no unmask capability

### Request Validation

- Input length limits (max 2000 chars)
- Parameter validation (k: 1-20, temperature: 0-1)
- Rate limiting per IP (configurable)

### Audit Trail

- Every request logged with:
  - Timestamp (UTC ISO format)
  - Route taken (answer/refusal)
  - PII masked flag
  - Client IP (last octet for privacy)
  - Latency

## Scalability

### Horizontal Scaling

- Stateless API (no session storage)
- Models loaded per-instance (no shared state)
- FAISS index read-only (safe for multiple processes)

**Deployment Pattern**:
```
Load Balancer
    ├── API Instance 1 (with models)
    ├── API Instance 2 (with models)
    └── API Instance 3 (with models)
```

### Performance Optimization

1. **Lazy Loading**: Models loaded on first use
2. **Caching**: Future: Redis for common queries
3. **Batching**: Future: Group requests for GPU efficiency
4. **Quantization**: int8/int4 for faster CPU inference

### Resource Requirements

**Minimum (CPU)**:
- 8 CPU cores
- 16 GB RAM
- 10 GB storage (models + index)

**Recommended (GPU)**:
- 8 CPU cores
- 16 GB RAM
- 8 GB VRAM (GPU)
- 20 GB storage

## Multilingual Architecture

### Language Detection

```python
if devanagari_chars > 3: return "HI"
if tamil_chars > 3: return "TA"
else: return "EN"
```

### Code-Mix Handling

**Problem**: "UPI ki limit kitna hai?" (Hinglish)

**Solution**:
1. Detect base language (EN due to low Devanagari)
2. Apply Hinglish synonyms: "kitna" → "how much"
3. Normalized: "UPI ki limit how much hai?"
4. Retrieve: Matches both EN and HI docs

### Multilingual Embedding

`paraphrase-multilingual-mpnet-base-v2` supports 50+ languages:
- Single embedding space
- Cross-lingual retrieval (EN query → HI docs)
- No language-specific indexes needed

## Fine-Tuning Architecture

### LoRA (Low-Rank Adaptation)

**Why LoRA**:
- Only train 1-2% of parameters
- Fast training (minutes on GPU)
- Small checkpoint (~10-50 MB)
- No catastrophic forgetting

**Target Modules** (Phi-3):
- q_proj, k_proj, v_proj, o_proj (attention layers)

**Training Pipeline**:
```
SFT Data (JSONL) → Format Prompts → LoRA Training → Adapter Weights
```

**Integration**:
```python
base_model = AutoModelForCausalLM.from_pretrained("phi-3-mini")
model = PeftModel.from_pretrained(base_model, "out/slm-lora")
```

## Testing Strategy

### Unit Tests
- `test_redaction.py`: PII patterns, scrubbing
- `test_language.py`: Detection, normalization
- `test_retrieval.py`: Index building, search
- `test_pipeline.py`: Route logic, citations
- `test_api.py`: Endpoints, validation

### Integration Tests
- End-to-end query flows
- Error handling
- Rate limiting

### Evaluation
- Recall@k, nDCG@k (retrieval quality)
- Citation coverage (generation quality)
- Refusal accuracy (safety)

## Future Enhancements

### Short-term
- [ ] Cross-encoder reranking
- [ ] Query caching (Redis)
- [ ] Conversation history

### Medium-term
- [ ] Hybrid search (keyword + semantic)
- [ ] Multi-turn dialogue
- [ ] Voice I/O

### Long-term
- [ ] Auto KB updates from FAQs
- [ ] Federated learning for privacy
- [ ] Multi-modal (image support)

## Monitoring in Production

### Key Metrics to Track

1. **Latency**: P50 < 2s, P95 < 5s (CPU)
2. **Throughput**: > 10 QPS baseline
3. **Refusal Rate**: Track for abuse patterns
4. **Citation Rate**: Should be > 90% for answers
5. **Error Rate**: < 1% target

### Alerting

- High latency (P95 > 10s)
- High error rate (> 5%)
- Model OOM failures
- FAISS index load failures

### Dashboards

- Grafana + Prometheus stack
- Visualize: QPS, latency heatmap, route distribution
- Audit log analysis for patterns

---

**Architecture Version**: 1.0  
**Last Updated**: 2025-01-15
