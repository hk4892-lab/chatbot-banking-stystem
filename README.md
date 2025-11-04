# BankBot – Multilingual Banking Chatbot

## Overview
BankBot is a production-oriented FastAPI service that delivers a multilingual (English, Tamil, Hindi) banking assistant. It combines lightweight Retrieval-Augmented Generation (TF-IDF over curated knowledge bases) with optional Phi-3-mini SLM completion, policy guardrails, and mock banking tools for EMI, card blocking, and interest-rate lookup. A small dark-mode web UI demonstrates the chat flow.

## Features
- FastAPI API with `/`, `/healthz`, and `/chat` (OpenAI-style messages payload)
- TF-IDF retrieval across English, Tamil, and Hindi knowledge bases
- Optional Phi-3-mini SLM generation (`USE_SLM=1`) with safe prompts and logging redaction
- Tool integrations: EMI calculator, card block ticket, and interest-rate lookup
- PII redaction for logs and escalation policy for sensitive requests
- Lightweight HTML/JS dark-mode chat client
- Pytest smoke tests for health and minimal chat

## Requirements
- Python 3.12 (recommended)
- pip / virtualenv tooling

## Setup
1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Configure environment:
   ```powershell
   copy .env.example .env
   # update values as needed
   ```
4. Run the API (reload for development):
   ```powershell
   .\run_api.bat
   ```
   Open http://127.0.0.1:8000/docs for Swagger UI.
5. (Optional) Launch the static UI demo:
   ```powershell
   .\run_ui_http.bat
   ```
   Visit http://127.0.0.1:5173 in your browser.

## Notes
- Set `USE_SLM=0` on low-memory CPUs to skip Phi-3 loading and rely on retrieval + policies.
- Optional packages (`sentence-transformers`, `bitsandbytes`) can be installed for future enhancements but are not required.
- Logs are sanitized with PII redaction and written to `bankbot/logs/audit.log` for monitoring.
- Run tests with `pytest -q` to ensure core functionality remains healthy.