#!/bin/bash
# Quick setup script for Banking SLM-RAG Chatbot

set -e

echo "🏦 Banking SLM-RAG Chatbot Setup"
echo "================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p index logs

echo "✓ Directories created"

# Copy environment config
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "✓ .env created (you can customize it)"
fi

# Build index
echo ""
echo "Building FAISS index from sample KB..."
python index/build_index.py data/kb_chunks.sample.jsonl index/kb.faiss index/kb.meta.pkl

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate venv:     source venv/bin/activate"
echo "  2. Start API:         make run-api"
echo "  3. Start UI:          make ui  (in another terminal)"
echo "  4. Run tests:         make test"
echo ""
echo "See README.md for full documentation."
echo "=========================================="
