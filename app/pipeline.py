"""End-to-end RAG pipeline: redact → retrieve → prompt → generate → cite."""
import re
import time
from typing import Dict, Literal

from app.redaction import redact, has_sensitive_keywords
from app.language import detect_lang, normalize, get_refusal_template
from app.retrieval import FAISSRetriever
from app.generator import SLMGenerator
from app.config import settings


class RAGPipeline:
    """Production RAG pipeline for banking chatbot."""

    def __init__(self, retriever: FAISSRetriever, generator: SLMGenerator):
        """
        Initialize pipeline with components.
        
        Args:
            retriever: FAISS retriever instance
            generator: SLM generator instance
        """
        self.retriever = retriever
        self.generator = generator

    def answer(
        self,
        text: str,
        k: int = 5,
        lang: Literal["AUTO", "EN", "HI", "TA"] = "AUTO",
        temperature: float = 0.3,
    ) -> Dict:
        """
        Process user query through full pipeline.
        
        Args:
            text: Raw user query
            k: Number of documents to retrieve
            lang: Language hint (AUTO for detection)
            temperature: Generation temperature
            
        Returns:
            Response dict with reply, citations, route, safety info, etc.
        """
        start_time = time.time()
        
        # Step 1: Redact PII
        masked_text, pii_mapping = redact(text)
        pii_masked = len(pii_mapping) > 0
        
        # Step 2: Detect and normalize language
        if lang == "AUTO":
            detected_lang = detect_lang(masked_text)
        else:
            detected_lang = lang
        
        normalized_text = normalize(masked_text, detected_lang)
        
        # Step 3: Check for sensitive/forbidden queries
        is_sensitive = has_sensitive_keywords(normalized_text)
        
        # Step 4: Retrieve relevant documents
        passages = self.retriever.search(normalized_text, k=k)
        
        # Step 5: Check if we have sufficient evidence
        has_relevant_context = False
        if passages:
            # Check if top passages meet threshold
            top_scores = [p.get("score", 0.0) for p in passages[:min(3, len(passages))]]
            avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
            has_relevant_context = avg_score >= settings.refusal_threshold
        
        # Step 6: Decide route
        if not has_relevant_context or is_sensitive:
            # Refusal route
            reply = get_refusal_template(detected_lang)
            route = "refusal"
            citations = []
            used_context_ids = []
            refusal = True
        else:
            # Answer route
            prompt = self.generator.make_prompt(normalized_text, passages, detected_lang)
            reply = self.generator.generate(
                prompt,
                temperature=temperature,
                top_p=settings.default_top_p,
                max_new_tokens=settings.default_max_tokens,
            )
            route = "answer"
            
            # Extract citations from reply
            citations = self._extract_citations(reply, passages)
            used_context_ids = [p["doc_id"] for p in passages]
            refusal = False
        
        # Step 7: Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "reply": reply,
            "citations": citations,
            "used_context_ids": used_context_ids,
            "route": route,
            "latency_ms": latency_ms,
            "lang": detected_lang,
            "safety": {
                "pii_masked": pii_masked,
                "refusal": refusal,
            },
        }

    def _extract_citations(self, text: str, passages: list[Dict]) -> list[Dict]:
        """
        Extract citation references from generated text.
        
        Args:
            text: Generated response
            passages: Retrieved passages
            
        Returns:
            List of citation dicts
        """
        citations = []
        seen = set()
        
        # Pattern: [Title (doc_id)]
        citation_pattern = re.compile(r"\[([^\]]+)\s+\(([^)]+)\)\]")
        matches = citation_pattern.findall(text)
        
        for title, doc_id in matches:
            if doc_id not in seen:
                citations.append({"doc_id": doc_id, "title": title.strip()})
                seen.add(doc_id)
        
        # Fallback: use top passages if no explicit citations
        if not citations and passages:
            for passage in passages[:3]:
                doc_id = passage.get("doc_id", "")
                title = passage.get("title", "")
                if doc_id and doc_id not in seen:
                    citations.append({"doc_id": doc_id, "title": title})
                    seen.add(doc_id)
        
        return citations
