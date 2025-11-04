"""FastAPI application exposing the chatbot endpoints."""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Dict

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_200_OK, HTTP_429_TOO_MANY_REQUESTS

from . import language
from .config import get_settings
from .deps import get_pipeline
from .pipeline import ChatPipeline
from .schemas import ChatTurnRequest, ChatTurnResponse, ConfigResponse, DebugPromptResponse


APP_TITLE = 'Banking SLM RAG Chatbot'
APP_DESCRIPTION = 'Production-grade multilingual banking support chatbot powered by SLM + RAG.'


REQUEST_COUNTER = Counter('chatbot_requests_total', 'Total number of requests', ['route', 'status'])
REQUEST_LATENCY = Histogram('chatbot_request_latency_seconds', 'Request latency', ['route'])
REFUSAL_COUNTER = Counter('chatbot_refusals_total', 'Total number of refusals', ['route'])


class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, *, rate: float, burst: int, key_fn: Callable[[Request], str]) -> None:
        super().__init__(app)
        self.rate = rate
        self.burst = burst
        self.key_fn = key_fn
        self.buckets: Dict[str, Dict[str, float]] = {}

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        key = self.key_fn(request)
        now = time.monotonic()
        bucket = self.buckets.get(key, {'tokens': float(self.burst), 'timestamp': now})
        elapsed = now - bucket['timestamp']
        bucket['tokens'] = min(self.burst, bucket['tokens'] + elapsed * self.rate)
        bucket['timestamp'] = now
        if bucket['tokens'] < 1:
            self.buckets[key] = bucket
            return PlainTextResponse('Too Many Requests', status_code=HTTP_429_TOO_MANY_REQUESTS)
        bucket['tokens'] -= 1
        self.buckets[key] = bucket
        response = await call_next(request)
        return response


def _client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return 'global'


settings = get_settings()

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version='1.0.0')

allowed_origins = ['http://localhost', 'http://localhost:3000', 'http://127.0.0.1']
if settings.app_env == 'production':
    # Restrict to localhost unless overridden via proxy.
    allowed_origins = ['http://localhost']

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=['POST', 'GET'],
    allow_headers=['*'],
)

app.add_middleware(
    RateLimiterMiddleware,
    rate=settings.rate_limit_rps,
    burst=settings.rate_limit_burst,
    key_fn=_client_ip,
)


@app.get('/health', status_code=HTTP_200_OK)
async def health() -> Dict[str, str]:
    return {'status': 'ok'}


@app.get('/config', response_model=ConfigResponse)
async def config() -> ConfigResponse:
    settings = get_settings()
    return ConfigResponse(
        embedding_model=settings.embedding_model,
        slm_model=settings.slm_model,
        languages=list(language.available_langs().keys()),
        default_k=settings.default_k,
        default_temperature=settings.default_temperature,
        refusal_threshold=settings.refusal_threshold,
        min_relevance=settings.min_relevance,
        rate_limit_rps=settings.rate_limit_rps,
        rate_limit_burst=settings.rate_limit_burst,
        app_env=settings.app_env,
    )


@app.post('/chat/turn', response_model=ChatTurnResponse)
async def chat_turn(
    payload: ChatTurnRequest,
    pipeline: ChatPipeline = Depends(get_pipeline),
) -> ChatTurnResponse:
    route_name = '/chat/turn'
    start_time = time.perf_counter()
    try:
        response_dict = await asyncio.to_thread(
            pipeline.answer,
            payload.text,
            k=payload.k,
            lang=payload.lang,
            temperature=payload.temperature,
        )
    except Exception as exc:  # pragma: no cover
        REQUEST_COUNTER.labels(route=route_name, status='error').inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency = time.perf_counter() - start_time
    REQUEST_COUNTER.labels(route=route_name, status='ok').inc()
    REQUEST_LATENCY.labels(route=route_name).observe(latency)
    if response_dict['route'] == 'refusal':
        REFUSAL_COUNTER.labels(route=route_name).inc()
    return ChatTurnResponse(**response_dict)


@app.get('/metrics')
async def metrics() -> PlainTextResponse:
    data = generate_latest()
    return PlainTextResponse(data.decode('utf-8'), media_type=CONTENT_TYPE_LATEST)


@app.get('/debug/prompt', response_model=DebugPromptResponse)
async def debug_prompt(pipeline: ChatPipeline = Depends(get_pipeline)) -> DebugPromptResponse:
    state = pipeline.debug_prompt()
    return DebugPromptResponse(
        prompt=state.get('prompt', ''),
        retrieved=state.get('retrieved', []),
        lang='',
        refusal=state.get('reason') is not None,
        reason=state.get('reason'),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):  # pragma: no cover
    REQUEST_COUNTER.labels(route=request.url.path, status=str(exc.status_code)).inc()
    return JSONResponse(status_code=exc.status_code, content={'detail': exc.detail})
