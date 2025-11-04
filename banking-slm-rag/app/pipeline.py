"""End-to-end conversational pipeline."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import Settings
from .generator import GroundedGenerator, extract_citations
from .language import detect_lang, normalize
from .redaction import redact, scrub_log
from .retrieval import FaissRetriever, RetrievalResult


LOGGER = logging.getLogger(__name__)

REFUSAL_MESSAGES = {
    'EN': 'I’m sorry, I cannot help with that. Please use official banking channels such as Mobile Banking, NetBanking, or our helpline.',
    'HI': 'क्षमा कीजिए, मैं इस अनुरोध में सहायता नहीं कर सकता। कृपया मोबाइल बैंकिंग, नेट बैंकिंग या आधिकारिक हेल्पलाइन का उपयोग करें।',
    'TA': 'மன்னிக்கவும், நான் அந்த கோரிக்கையில் உதவ முடியாது. மொபைல் வங்கி, நெட் வங்கி அல்லது அதிகாரப்பூர்வ உதவி எண்ணை பயன்படுத்தவும்.',
}

SENSITIVE_KEYWORDS = (
    'my balance',
    'account balance',
    'otp',
    'one time password',
    'cvv',
    'pin number',
    'send otp',
    'tell me balance',
)


def _needs_refusal(text: str) -> Optional[str]:
    lowered = text.lower()
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in lowered:
            return f'sensitive keyword: {keyword}'
    return None


@dataclass
class PipelineState:
    last_prompt: Optional[str] = None
    last_retrieved: List[str] = None  # type: ignore[assignment]
    last_reason: Optional[str] = None


class ChatPipeline:
    """Core orchestrator for redact → retrieve → generate workflow."""

    def __init__(
        self,
        settings: Settings,
        retriever: FaissRetriever,
        generator: GroundedGenerator,
        *,
        audit_logger: Optional[logging.Logger] = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.audit_logger = audit_logger or LOGGER
        self.state = PipelineState(last_retrieved=[], last_prompt=None, last_reason=None)

    def answer(
        self,
        text: str,
        *,
        k: Optional[int] = None,
        lang: str = 'AUTO',
        temperature: Optional[float] = None,
    ) -> Dict[str, object]:
        start = time.perf_counter()
        settings = self.settings
        request_k = k or settings.default_k
        request_temperature = temperature if temperature is not None else settings.default_temperature

        masked_text, pii_mapping = redact(text)
        requested_lang = lang
        resolved_lang = detect_lang(text) if lang == 'AUTO' else lang

        refusal_reason = _needs_refusal(masked_text)

        normalized_query = normalize(masked_text, resolved_lang)
        retrieval_results: List[RetrievalResult] = []
        if not refusal_reason:
            retrieval_results = self.retriever.search(normalized_query, request_k)

        filtered = [res for res in retrieval_results if res.score >= settings.min_relevance]
        top_score = filtered[0].score if filtered else (retrieval_results[0].score if retrieval_results else 0.0)

        is_low_confidence = top_score < settings.refusal_threshold or not filtered
        route = 'refusal' if refusal_reason or is_low_confidence else 'answer'

        citations_payload: List[Dict[str, str]] = []
        used_ids: List[str] = []
        context_chunks = [result.chunk for result in filtered]

        if route == 'refusal':
            reply = REFUSAL_MESSAGES.get(resolved_lang, REFUSAL_MESSAGES['EN'])
        else:
            prompt, rendered_context = self.generator.make_prompt(masked_text, context_chunks, resolved_lang)
            generated = self.generator.generate(
                prompt,
                temperature=request_temperature,
                top_p=0.9,
                max_new_tokens=256,
            )
            citations = extract_citations(generated)
            if not citations:
                citations = [(chunk.title, chunk.doc_id) for chunk in context_chunks[:request_k]]
            citations_payload = [
                {'doc_id': doc_id, 'title': title}
                for title, doc_id in citations
            ]
            used_ids = [chunk.doc_id for chunk in context_chunks[:request_k]]
            reply = generated
            self.state.last_prompt = prompt
            self.state.last_retrieved = rendered_context
            self.state.last_reason = None

        if route == 'refusal':
            self.state.last_prompt = None
            self.state.last_retrieved = [f'{res.chunk.title} ({res.chunk.doc_id})' for res in filtered or retrieval_results]
            self.state.last_reason = refusal_reason or 'low relevance'

        latency_ms = int((time.perf_counter() - start) * 1000)

        audit_record = {
            'request_lang': requested_lang,
            'resolved_lang': resolved_lang,
            'masked_text': masked_text,
            'pii_tokens': list(pii_mapping.keys()),
            'route': route,
            'reason': refusal_reason if refusal_reason else ('low_confidence' if is_low_confidence else 'grounded'),
            'latency_ms': latency_ms,
            'top_score': top_score,
            'used_ids': used_ids,
        }
        sanitized = scrub_log(audit_record)
        self.audit_logger.info(json.dumps(sanitized, ensure_ascii=False))

        response = {
            'reply': reply,
            'citations': citations_payload,
            'used_context_ids': used_ids,
            'route': route,
            'latency_ms': latency_ms,
            'lang': resolved_lang,
            'safety': {
                'pii_masked': bool(pii_mapping),
                'refusal': route == 'refusal',
            },
        }
        return response

    def debug_prompt(self) -> Dict[str, object]:
        return {
            'prompt': self.state.last_prompt or '',
            'retrieved': self.state.last_retrieved or [],
            'reason': self.state.last_reason,
        }
