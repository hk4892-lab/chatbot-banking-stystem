# Banking SLM + RAG Chatbot

Multilingual, production-ready banking support assistant powered by Retrieval-Augmented Generation (RAG) with a small language model (SLM). The system delivers grounded, citation-rich answers in English, Hindi, and Tamil while enforcing strict PII redaction and refusal policies.

```
User → Redaction → Language Normalize → FAISS Retrieval → Grounded Prompt → SLM → Response + Citations
                ↑                                                      │
                └───────────── Metrics & Audit Logs (PII-scrubbed) ─────┘
```

## Features
- Deterministic RAG pipeline using FAISS (`IndexFlatIP`) with SentenceTransformers embeddings (`paraphrase-multilingual-mpnet-base-v2`).
- Grounded generation via `microsoft/phi-3-mini-4k-instruct` (override via `.env` for lighter local models) with LoRA fine-tuning hooks.
- Multilingual normalization and synonyms to handle Hinglish/Tamlish code-mix inputs.
- Deterministic PII masking for phones, cards, Aadhaar/PAN, and emails before retrieval or logging.
- FastAPI backend with `/chat/turn`, `/health`, `/config`, `/metrics`, `/debug/prompt` and Prometheus instrumentation.
- Streamlit operator console for local exploration.
- JSONL knowledge base & SFT datasets, TRL+PEFT training script, retrieval CLI, and evaluation harness.
- Dockerized deployment and Makefile automation.

## Quickstart

```bash
git clone <repo>
cd banking-slm-rag
make venv
source .venv/bin/activate
make index            # builds FAISS index from sample KB
make run-api          # starts FastAPI on :8000
make ui               # optional Streamlit console
```

### Env Vars (`.env.example`)

```
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
SLM_MODEL=microsoft/phi-3-mini-4k-instruct
FAISS_INDEX_PATH=index/kb.faiss
FAISS_META_PATH=index/kb.meta.pkl
DEFAULT_K=5
DEFAULT_TEMPERATURE=0.3
RATE_LIMIT_RPS=2
RATE_LIMIT_BURST=4
APP_ENV=development
AUDIT_LOG_PATH=logs/audit.log
```

Adjust `SLM_MODEL` to a lightweight checkpoint (e.g. `sshleifer/tiny-gpt2`) for local tests; the production default remains Phi-3 Mini.

## Make Targets
- `make venv` – create virtualenv & install dependencies.
- `make index` – build FAISS index with sample KB.
- `make run-api` – launch FastAPI (reload enabled).
- `make ui` – run Streamlit console.
- `make test` – run pytest suite.
- `make lint` – run Ruff & mypy.
- `make train-lora` – fine-tune SLM adapters with TRL + LoRA.

## Safety & Observability
- `app/redaction.py` masks PII before retrieval or logging; audit logs (`logs/audit.log`) contain only masked tokens.
- Refusal policy blocks balance/OTP/personal requests and low-confidence answers.
- Prometheus metrics via `/metrics`: request counts, latencies, refusal totals.
- JSONL audit logging for every turn (scrubbed), plus optional `/debug/prompt` to inspect masked prompts.

## Extending the KB
1. Append new records to `data/kb_chunks.sample.jsonl` (or maintain a separate JSONL file).
2. Run `make index` (or `python index/build_index.py <kb.jsonl> <index.faiss> <meta.pkl>`).
3. Redeploy or restart the API to load refreshed index.

## Fine-Tuning Workflow
Sample SFT datasets live under `data/`. Use `train/sft_lora.py` to train LoRA adapters:

```bash
make train-lora
```

This produces `out/slm-lora/`. Merge adapters or load them dynamically in `GroundedGenerator` for production.

## Evaluation
Run `python eval/evaluate.py --prompts eval/prompts.sample.csv --k 5` to compute Recall@k, nDCG@k, and grounded sentence ratio. See `eval/metrics.md` for metric definitions.

## Testing & CI
- `pytest` covers redaction, retrieval alignment with Hinglish/Tamlish queries, pipeline refusals, and API schema.
- Fakes replace heavyweight models during tests while the production code path continues to rely on SentenceTransformers + Phi-3 Mini.

## Docker

```bash
docker build -t banking-slm-rag .
docker run -p 8000:8000 --env-file .env banking-slm-rag
```

The container launches `uvicorn app.api:app` listening on port 8000.
