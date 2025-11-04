## BankBot – Multilingual Banking Chatbot

### Overview
BankBot is a production-ready FastAPI service that delivers a concise and safe multilingual (English, Tamil, Hindi) banking assistant. It combines multilingual sentence-transformer embeddings, a FAISS (with NumPy fallback) vector store, and an optional Phi-3-mini small language model to answer customer queries while respecting banking safety policies and redacting PII. A tiny dark-mode web UI demonstrates end-to-end chat flow.

### Features
- FastAPI backend with `/`, `/healthz`, and `/chat` endpoints plus permissive CORS for rapid prototyping
- Retrieval-Augmented Generation powered by multilingual embeddings and FAISS, automatically falling back to pure NumPy cosine search if FAISS is unavailable
- Optional Phi-3-mini SLM generation (`USE_SLM=1`) that consumes RAG context when retrieval scores are weak
- Banking policy layer with tool routing (EMI calculator, card block ticket, interest rate lookup) and escalation for sensitive actions
- Regex-based PII redaction before logging or returning responses
- Lightweight HTML/JS dark-mode chat client (`bankbot/app/ui/web/index.html`)
- Windows-friendly scripts (`run_api.bat`, `run_ui_http.bat`) and pytest smoke coverage

### Prerequisites
- Python 3.12
- git, pip, and virtualenv tooling

### Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```
Update `.env` as needed (model IDs, ports, thresholds) and then launch:

```powershell
powershell -File run_api.bat      # FastAPI server -> http://127.0.0.1:8000/docs
powershell -File run_ui_http.bat  # Static UI -> http://127.0.0.1:5173
```

Run tests anytime with:
```powershell
pytest -q
```

### Notes
- Retrieval exclusively uses multilingual embeddings + FAISS; TF-IDF is not used anywhere.
- When switching to an `intfloat/multilingual-e5-*` embedder, remember the required "query:" / "passage:" prefixes (handled automatically in code).
- Set `USE_SLM=0` on low-RAM CPUs to skip loading Phi-3 and rely entirely on fast RAG responses.
- Vector indexes persist under `bankbot/app/data/index/` for quick restarts; delete the folder to rebuild.

### Troubleshooting
- **GET /** returns 404 -> ensure the FastAPI process is running (`uvicorn bankbot.app.api:app`) and that you are using the correct `HOST` / `PORT` values from `.env`.
- **Slow CPU responses** -> disable the SLM by setting `USE_SLM=0`; the retrieval path remains fully functional.
- **CORS errors** when hosting the UI elsewhere -> adjust the FastAPI CORS configuration or proxy the API so the UI origin is allowed.
