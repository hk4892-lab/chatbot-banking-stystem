.PHONY: venv run-api ui index test lint train-lora clean help

PYTHON=python3
VENV=venv
BIN=$(VENV)/bin

help:
	@echo "Banking SLM-RAG Chatbot - Makefile Commands"
	@echo "==========================================="
	@echo "make venv         - Create virtual environment and install dependencies"
	@echo "make run-api      - Start FastAPI server (dev mode)"
	@echo "make ui           - Start Streamlit UI"
	@echo "make index        - Build FAISS index from KB chunks"
	@echo "make test         - Run pytest suite"
	@echo "make lint         - Run ruff and mypy"
	@echo "make train-lora   - Fine-tune SLM with LoRA"
	@echo "make clean        - Remove generated files"

venv:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@echo "Virtual environment ready. Activate with: source $(VENV)/bin/activate"

run-api:
	$(BIN)/uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

ui:
	$(BIN)/streamlit run ui/app_streamlit.py

index:
	mkdir -p index logs
	$(BIN)/python index/build_index.py data/kb_chunks.sample.jsonl index/kb.faiss index/kb.meta.pkl

test:
	$(BIN)/pytest -q tests/

lint:
	$(BIN)/ruff check app/ ui/ index/ train/ eval/ tests/
	$(BIN)/mypy app/ --ignore-missing-imports

train-lora:
	mkdir -p out/slm-lora
	$(BIN)/python train/sft_lora.py data/sft_train.sample.jsonl data/sft_val.sample.jsonl out/slm-lora

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf index/*.faiss index/*.pkl out/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
