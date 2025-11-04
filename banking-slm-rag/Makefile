PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: venv run-api ui index test lint train-lora clean

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run-api:
	$(PY) -m uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

ui:
	$(PY) -m streamlit run ui/app_streamlit.py

index:
	$(PY) index/build_index.py data/kb_chunks.sample.jsonl index/kb.faiss index/kb.meta.pkl

test:
	$(PY) -m pytest -q

lint:
	$(VENV)/bin/ruff check .
	$(PY) -m mypy app

train-lora:
	$(PY) train/sft_lora.py data/sft_train.sample.jsonl data/sft_val.sample.jsonl out/slm-lora

clean:
	rm -rf $(VENV) __pycache__ */__pycache__ .pytest_cache .mypy_cache
