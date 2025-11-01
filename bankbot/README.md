# BankBot

BankBot is a multilingual banking support chatbot that runs locally with a FastAPI backend and a Streamlit chat UI. It understands English, Hindi (Devanagari), and Tamil queries, handles code-mixed Hinglish/Tanglish phrases, redacts personal information deterministically before processing, and retrieves answers from a curated knowledge base with TF-IDF (and optional sentence-transformers) retrieval. Tool calls cover EMI calculation, card blocking, and rate lookups, with responses fully audited in a JSONL log.

## Features
- Multilingual and code-mixed support (EN/HI/TA) via Unicode script detection and synonym expansion
- Deterministic PII redaction (phone, email, PAN, card, account, Aadhaar) with reversible tokens kept server-side only
- Retrieval-augmented responses with "evidence-or-silence" gating and inline citations
- Deterministic routing policy (answer, ask, tool, escalate) with session memory and audit logging
- Streamlit dark-mode chat UI with language toggle, confidence slider, sentence-transformer option, quick actions, and safety banner
- FastAPI backend with `/chat/turn`, `/healthz`, and `/config` endpoints, plus JSONL audit log and CLI viewer
- Pytest coverage for redaction, retrieval, and policy behavior

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api:app --reload
streamlit run ui/app_streamlit.py
```

### Optional sentence-transformers
Install `sentence-transformers` and enable the checkbox in the UI or pass `"use_sentence_transformers": true` in API calls.

```bash
pip install sentence-transformers
```

## Project Layout

```
bankbot/
  app/
    api.py               # FastAPI app and audit logging
    core/                # language, redaction, retrieval, policy, rendering
    tools/               # calculate_emi, block_card, fetch_rate
    data/                # multilingual KB JSON files
  ui/app_streamlit.py    # Streamlit chat UI (dark mode)
  scripts/               # evaluate.py and show_logs.py utilities
  tests/                 # pytest suites for safety and routing
  logs/                  # runtime audit log location (JSONL)
  requirements.txt       # runtime dependencies
  pyproject.toml         # metadata + pytest config
  ruff.toml              # linting defaults
  .streamlit/config.toml # dark theme settings
```

## Knowledge Base
- English, Hindi, and Tamil KB files live under `app/data/`
- Each entry includes `id`, `title`, `content`, and `tags`
- To add entries, append JSON objects to the respective language file and restart the backend to rebuild the index

## Testing & Quality

```bash
pytest -q
```

- `tests/test_redaction.py`: verifies deterministic tokenization and ensures raw PII never reaches the audit log
- `tests/test_retrieval.py`: checks Hinglish query coverage and score thresholds
- `tests/test_policy.py`: validates routing for escalation and EMI tool invocation
- Ruff config is provided; run `ruff check .` if installed

## Evaluation Script

```bash
python scripts/evaluate.py --prompts scripts/prompts.csv
```

Outputs top-1 accuracy vs. the expected substring, refusal rate, and average confidence. Add `--use-sentence-transformers` to benchmark with the optional reranker. The prompt CSV includes English, Hindi, Tamil, and code-mixed examples.

## Logs & Observability
- Every `/chat/turn` writes to `logs/audit.log` in JSONL `{ts, lang, redaction_tokens, route, topdoc_id, top_score, tools_called, ...}` format
- View recent entries with:

```bash
python scripts/show_logs.py --tail 50
```

Tokens (not raw values) are recorded so the audit trail never exposes PII.

## Safety Notes & Limitations
- Demo only; no real banking operations are performed
- Do not share real personal data: although redaction is in place, this preview is intended for local experimentation
- Knowledge base is concise and may not cover every scenario; low-confidence queries politely escalate
- Session memory is in-memory only and resets when the process restarts

## Example Queries
- English: "How do I update my KYC documents?"
- Hindi: "?????? ???? ????? ???"
- Tamil: "?????? ??? ?????? ?????? ?????? ????????"
- Hinglish: "statement send pannunga"
- Tanglish: "emi calc pannunga 5 lakh 10% 60 months"

## Disclaimer
This is a demo for illustrative purposes. No actual banking transactions are executed, and the responses are generated from a static knowledge base. Always consult official bank channels for real support.
