# Evaluation Metrics

This document describes the evaluation metrics used to assess the Banking SLM-RAG Chatbot.

## Retrieval Metrics

### Recall@k

**Definition**: Proportion of queries where the expected document appears in the top-k retrieved results.

**Formula**: `Recall@k = 1` if expected doc in top-k, else `0`

**Interpretation**:
- 1.0 = Perfect (expected doc always retrieved)
- 0.8 = Good (80% of queries retrieve relevant doc)
- < 0.5 = Poor (retrieval needs improvement)

**Target**: > 0.85 for k=5

### nDCG@k (Normalized Discounted Cumulative Gain)

**Definition**: Measures ranking quality, giving higher scores when relevant docs appear earlier.

**Formula**: `nDCG@k = DCG@k / IDCG@k`
- DCG considers position: earlier = better
- Normalized against ideal ranking

**Interpretation**:
- 1.0 = Perfect ranking (relevant doc at position 1)
- 0.7-0.9 = Good ranking
- < 0.5 = Poor ranking

**Target**: > 0.80 for k=5

## Generation Metrics

### Citation Coverage

**Definition**: Percentage of responses that include proper citations in format `[Title (doc_id)]`.

**Measurement**: Regex pattern matching on generated responses.

**Target**: > 90% for answer route

### Groundedness

**Definition**: Whether the response is supported by retrieved context.

**Measurement** (manual or LLM-based):
- Extract claims from response
- Check if each claim appears in retrieved docs
- Score = % of grounded claims

**Target**: > 95% grounded claims

## Safety Metrics

### PII Redaction Rate

**Definition**: Percentage of PII entities successfully masked.

**Measurement**: Test with known PII patterns, verify masking.

**Target**: 100% for known patterns

### Refusal Accuracy

**Definition**: Percentage of forbidden queries (balance, transactions, OTP) that are correctly refused.

**Measurement**: Test with forbidden query dataset.

**Target**: 100% refusal on forbidden queries

## Performance Metrics

### Latency (P50, P95)

**Definition**: Response time distribution.

**Target**:
- P50 < 2 seconds (CPU), < 500ms (GPU)
- P95 < 5 seconds (CPU), < 2 seconds (GPU)

### Throughput

**Definition**: Queries per second.

**Target**: > 10 QPS (depends on hardware)

## Running Evaluation

```bash
# Build index first
make index

# Run evaluation
python eval/evaluate.py eval/prompts.sample.csv 5

# Expected output:
# Recall@5: > 0.85
# nDCG@5: > 0.80
```

## Interpreting Results

### Good Results
- Recall@5 > 0.85
- nDCG@5 > 0.80
- All refusal queries correctly refused
- Citations present in > 90% of answers

### Poor Results & Fixes

**Low Recall**:
- Issue: Relevant docs not retrieved
- Fix: Improve embedding model, add more KB content, tune normalization

**Low nDCG**:
- Issue: Relevant docs appear late in ranking
- Fix: Better embeddings, re-rank with cross-encoder

**Missing Citations**:
- Issue: Model not citing sources
- Fix: Fine-tune with more citation examples, adjust system prompt

**False Refusals**:
- Issue: Valid queries refused
- Fix: Lower refusal threshold, improve keyword detection

**Missed Refusals**:
- Issue: Forbidden queries answered
- Fix: Strengthen keyword list, add has_sensitive_keywords checks

## Continuous Monitoring

In production, track:
1. **Daily metrics**: Latency, throughput, error rate
2. **Weekly review**: Refusal rate, citation coverage
3. **Monthly audit**: Sample 100 conversations for quality
4. **User feedback**: Thumbs up/down on responses
