# 🚀 Quickstart Guide

Get the Banking SLM-RAG Chatbot running in under 5 minutes!

## Prerequisites

- Python 3.11+
- 16 GB RAM minimum
- 10 GB free disk space
- Internet connection (for downloading models)

## Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Start the API
make run-api
```

Open browser: http://localhost:8000/docs

## Option 2: Manual Setup

### Step 1: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit if needed (optional)
nano .env
```

### Step 3: Build FAISS Index

```bash
# Create directories
mkdir -p index logs

# Build index from sample KB
python index/build_index.py \
  data/kb_chunks.sample.jsonl \
  index/kb.faiss \
  index/kb.meta.pkl
```

Expected output:
```
Encoding 30 chunks...
Saved FAISS index to index/kb.faiss
Saved metadata to index/kb.meta.pkl
```

### Step 4: Start API Server

```bash
# Start FastAPI server
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

### Step 5: Test It

```bash
# In another terminal
curl http://localhost:8000/health

# Try a query
curl -X POST http://localhost:8000/chat/turn \
  -H "Content-Type: application/json" \
  -d '{
    "text": "How to reset ATM PIN?",
    "k": 5,
    "lang": "AUTO",
    "temperature": 0.3
  }'
```

## Option 3: Docker

```bash
# Build image
docker build -t banking-chatbot .

# Build index first (outside container)
make index

# Run container
docker run -d \
  --name banking-chatbot \
  -p 8000:8000 \
  -v $(pwd)/index:/app/index \
  -v $(pwd)/logs:/app/logs \
  banking-chatbot:latest

# Check logs
docker logs -f banking-chatbot
```

## Using the UI

```bash
# In a new terminal (with venv activated)
streamlit run ui/app_streamlit.py
```

Open browser: http://localhost:8501

## Running Tests

```bash
# All tests
make test

# Specific test file
pytest tests/test_redaction.py -v
```

## Example Queries

Try these in the UI or API:

**English**:
- "How to reset ATM PIN?"
- "What is UPI transaction limit?"
- "How to block my debit card?"

**Hindi**:
- "एटीएम पिन कैसे रीसेट करें?"
- "यूपीआई की सीमा क्या है?"

**Tamil**:
- "எப்படி கார்டை பிளாக் செய்வது?"

**Hinglish**:
- "UPI ki limit kitna hai?"
- "Card block kaise karein?"

**Forbidden (should refuse)**:
- "Show my balance"
- "Transfer money"

## Troubleshooting

### "Module not found" error
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### "Index not found" error
```bash
# Rebuild index
make index
```

### Out of memory
```bash
# Use smaller model
export SLM_MODEL=microsoft/phi-2
```

### Slow responses
```bash
# First query loads models (takes 30-60s)
# Subsequent queries are faster (~2-5s on CPU)
```

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Explore [eval/metrics.md](eval/metrics.md) for evaluation
- Review [train/README.md](train/README.md) for fine-tuning

## Need Help?

- Check logs: `tail -f logs/audit.jsonl`
- View metrics: http://localhost:8000/metrics
- API docs: http://localhost:8000/docs
- Debug prompt: http://localhost:8000/debug/prompt?text=test

---

**🎉 You're all set! Happy chatting!**
