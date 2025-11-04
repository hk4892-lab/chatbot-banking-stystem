"""FastAPI application with chat, health, config, and metrics endpoints."""
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.config import settings
from app.schemas import (
    ChatTurnRequest,
    ChatTurnResponse,
    Citation,
    SafetyInfo,
    HealthResponse,
    ConfigResponse,
)
from app.deps import get_pipeline, write_audit_log
from app.redaction import scrub_log

# Initialize FastAPI app
app = FastAPI(
    title="Banking SLM-RAG Chatbot",
    description="Production-grade banking support chatbot using SLM + RAG",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
request_counter = Counter(
    "chatbot_requests_total",
    "Total number of requests",
    ["endpoint", "route"],
)

latency_histogram = Histogram(
    "chatbot_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

refusal_counter = Counter(
    "chatbot_refusals_total",
    "Total number of refusals",
)

# Rate limiting (simple in-memory)
rate_limit_store: Dict[str, List[float]] = defaultdict(list)


def check_rate_limit(client_id: str) -> bool:
    """
    Check if client exceeds rate limit.
    
    Args:
        client_id: Client identifier (IP or user ID)
        
    Returns:
        True if within limit, False if exceeded
    """
    if not settings.rate_limit_enabled:
        return True
    
    now = time.time()
    window_start = now - settings.rate_limit_window_seconds
    
    # Clean old timestamps
    rate_limit_store[client_id] = [
        ts for ts in rate_limit_store[client_id] if ts > window_start
    ]
    
    # Check limit
    if len(rate_limit_store[client_id]) >= settings.rate_limit_requests:
        return False
    
    # Add current request
    rate_limit_store[client_id].append(now)
    return True


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get configuration information."""
    return ConfigResponse(
        embedding_model=settings.embedding_model,
        slm_model=settings.slm_model,
        default_k=settings.default_k,
        default_temperature=settings.default_temperature,
        languages=["EN", "HI", "TA"],
        refusal_threshold=settings.refusal_threshold,
    )


@app.post("/chat/turn", response_model=ChatTurnResponse)
async def chat_turn(request: ChatTurnRequest, req: Request):
    """
    Handle chat turn with RAG pipeline.
    
    Args:
        request: Chat turn request
        req: FastAPI request object
        
    Returns:
        Chat turn response with answer/refusal and metadata
    """
    start_time = time.time()
    
    # Rate limiting
    client_ip = req.client.host if req.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Get pipeline
    pipeline = get_pipeline()
    
    # Process query
    try:
        result = pipeline.answer(
            text=request.text,
            k=request.k,
            lang=request.lang,
            temperature=request.temperature,
        )
    except Exception as e:
        # Log error and return graceful failure
        write_audit_log({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "error",
            "error": str(e),
            "query": scrub_log({"text": request.text}),
        })
        raise HTTPException(status_code=500, detail="Internal processing error")
    
    # Build response
    response = ChatTurnResponse(
        reply=result["reply"],
        citations=[Citation(**c) for c in result["citations"]],
        used_context_ids=result["used_context_ids"],
        route=result["route"],
        latency_ms=result["latency_ms"],
        lang=result["lang"],
        safety=SafetyInfo(**result["safety"]),
    )
    
    # Update metrics
    request_counter.labels(endpoint="/chat/turn", route=result["route"]).inc()
    latency_histogram.labels(endpoint="/chat/turn").observe(time.time() - start_time)
    
    if result["route"] == "refusal":
        refusal_counter.inc()
    
    # Audit log (scrubbed)
    audit_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": "chat_turn",
        "route": result["route"],
        "lang": result["lang"],
        "k": request.k,
        "temperature": request.temperature,
        "latency_ms": result["latency_ms"],
        "citations_count": len(result["citations"]),
        "pii_masked": result["safety"]["pii_masked"],
        "client_ip": client_ip,
    }
    write_audit_log(scrub_log(audit_record))
    
    return response


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/debug/prompt")
async def debug_prompt(text: str, k: int = 5):
    """
    Debug endpoint to see the masked prompt.
    
    Args:
        text: Query text
        k: Number of documents to retrieve
        
    Returns:
        Masked prompt for inspection
    """
    from app.redaction import redact
    from app.language import detect_lang, normalize
    
    pipeline = get_pipeline()
    
    # Redact and normalize
    masked_text, _ = redact(text)
    lang = detect_lang(masked_text)
    normalized_text = normalize(masked_text, lang)
    
    # Retrieve
    passages = pipeline.retriever.search(normalized_text, k=k)
    
    # Build prompt
    prompt = pipeline.generator.make_prompt(normalized_text, passages, lang)
    
    return {
        "original": text,
        "masked": masked_text,
        "normalized": normalized_text,
        "lang": lang,
        "retrieved_count": len(passages),
        "prompt": prompt,
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Warm up models on startup (optional)."""
    # Uncomment to preload models
    # get_pipeline()
    pass
