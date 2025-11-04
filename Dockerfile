# Multi-stage build for Banking SLM-RAG Chatbot
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /build/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY index/ ./index/
COPY .env.example .env

# Create necessary directories
RUN mkdir -p logs

# Pre-build index (optional, can be mounted as volume)
# RUN python index/build_index.py data/kb_chunks.sample.jsonl index/kb.faiss index/kb.meta.pkl

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run API server
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
