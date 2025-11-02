# BankBot ? Multilingual Banking Support Chatbot

BankBot is a production-ready demo that showcases a multilingual (English, Hindi, Tamil) banking support assistant. It understands code-mixed queries, performs safe PII redaction, retrieves grounded answers from a curated knowledge base, and surfaces deterministic tool actions such as EMI calculation, card blocking, and rate lookups. The backend is powered by FastAPI, the frontend by Streamlit, and everything runs locally without external services.

## Highlights

- **Multilingual understanding** with script detection (Devanagari, Tamil, Latin) plus synonym rewrites for common Hinglish/Tanglish banking phrases.
- **Deterministic PII redaction** that replaces sensitive values with reversible tokens before any retrieval or logging takes place.
- **Retrieval-Augmented Generation** using a TF-IDF baseline with optional SentenceTransformer re-ranking when available.
- **Evidence-or-silence policy** that refuses when confidence is low, asks clarifying questions, or escalates on demand.
- **Auditable operations**: every turn is written to `logs/audit.log` (tokens only), and a CLI helper pretty-prints the latest entries.
- **Tooling** for EMI calculation, card blocking, and rate lookup with safe detokenisation strictly inside tool execution.

## Safety Notes & Limitations

- The bot stores only tokenised PII in audit logs and never echoes raw sensitive values back to the UI.
- This is a demo environment; it does **not** perform actual banking operations or network calls beyond the local server.
- Knowledge base content is intentionally generic. Update or expand it before using the bot in a different context.
- SentenceTransformer support is optional and requires the package to be installed manually (`pip install sentence-transformers`).

## Quickstart

```bash
cd bankbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the FastAPI backend
uvicorn app.api:app --reload

# In a separate terminal (with the same virtualenv) run the Streamlit UI
streamlit run ui/app_streamlit.py
```

The Streamlit app listens on `http://localhost:8501` and communicates with the FastAPI backend at `http://localhost:8000`.

## Project Structure

- `app/` ? FastAPI app plus core modules for language handling, redaction, retrieval, policy, and tools.
- `ui/` ? Streamlit single-page chat interface (dark mode ready).
- `app/data/` ? Knowledge bases for English, Hindi, and Tamil.
- `logs/` ? Runtime JSONL audit logs (created automatically).
- `scripts/` ? Evaluation helper and log viewer.
- `tests/` ? Pytest suite covering redaction, retrieval, and policy routing.

## Configuration & Tuning

- **Threshold slider**: adjust the evidence gate (default `0.18`) in the Streamlit sidebar.
- **Language toggle**: force English/Hindi/Tamil or let the system auto-detect the dominant script.
- **Sentence transformers**: install `sentence-transformers` and enable the checkbox in the UI (or pass `use_sentence_transformers=true` in API calls).

To add new knowledge base entries, extend the JSON files under `app/data/`. Keep each item in the format:

```json
{
  "id": "kb_en_011",
  "title": "Short headline",
  "content": "2-3 sentence explanation.",
  "tags": ["keyword", "another"]
}
```

After editing, restart the backend so the retrieval index rebuilds.

## Example Queries

- `UPI limit kitna hai?`
- `Need debit card block steps`
- `statement send pannunga`
- `???????? ?????? bill due date?`
- `calculate emi P=500000 r=10 n=60`

## Evaluation & Observability

- Run the lightweight retrieval evaluation:

  ```bash
  python scripts/evaluate.py scripts/prompts.csv
  ```

- Tail recent audit logs in a readable format:

  ```bash
  python scripts/show_logs.py --tail 50
  ```

## Testing & Linting

```bash
pytest -q
ruff check .
```

## API Contract

- `POST /chat/turn` ? `{message}` in, `{lang, route, answer, citations, tool_result, confidence}` out.
- `GET /healthz` ? basic readiness.
- `GET /config` ? current server configuration (threshold, languages, embedding availability).

Refer to `ui/app_streamlit.py` for example client interactions.

## Disclaimer

This repository is for demonstration purposes only. It is **not** a production banking system and performs no live financial transactions. Always consult real banking systems and security teams before handling sensitive data in production.