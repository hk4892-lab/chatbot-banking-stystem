"""Grounded generation using a small language model."""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .config import Settings
from .retrieval import KnowledgeChunk


LOGGER = logging.getLogger(__name__)


class GroundedGenerator:
    """Wrapper around a causal LM for grounded response generation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self._resolve_device(settings.device_preference)
        self.tokenizer = None
        self.model = None
        self._load_model()

    @staticmethod
    def _resolve_device(preference: str) -> str:
        if preference == 'cpu':
            return 'cpu'
        if preference == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def _load_model(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.settings.slm_model, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.slm_model,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto' if self.device == 'cuda' else None,
            )
            if self.device == 'cpu':
                self.model.to('cpu')
            self.model.eval()
        except Exception as exc:  # pragma: no cover - exercised in integration tests
            LOGGER.warning('Falling back to templated generator due to load error: %s', exc)
            self.model = None
            self.tokenizer = None

    def make_prompt(self, query: str, passages: Iterable[KnowledgeChunk], lang: str) -> Tuple[str, List[str]]:
        context_lines: List[str] = []
        for chunk in passages:
            context_lines.append(f"[{chunk.title} ({chunk.doc_id})]\n{chunk.content}")
        context_block = '\n\n'.join(context_lines)
        system_message = (
            'System: Answer strictly from CONTEXT. If information is missing, refuse politely '
            'and direct the user to official channels. Always cite as [Title (doc_id)]. Never unmask PII.'
        )
        prompt = (
            f"{system_message}\n\nCONTEXT:\n{context_block}\n\n"
            f"User ({lang}): {query}\nAssistant:"
        )
        return prompt, context_lines

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            return self._fallback_generate(prompt)
        input_ids = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.settings.max_prompt_context_tokens).input_ids
        input_ids = input_ids.to(self.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, generation_config=generation_config)
        generated = outputs[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            return self._fallback_generate(prompt)
        return text

    def _fallback_generate(self, prompt: str) -> str:
        """Template-based response used when the SLM is unavailable."""

        lines = [line for line in prompt.splitlines() if line.startswith('[')]
        if not lines:
            return 'I’m sorry, I cannot help with that. Please contact the official banking help desk.'
        citations = [line.split(']')[0].strip('[') for line in lines[:2]]
        sentences = []
        for citation in citations:
            sentences.append(f'Refer to [{citation}] for the requested information.')
        sentences.append('For more help please use the official banking channels.')
        return ' '.join(sentences)


CITATION_PATTERN = re.compile(r'\[(?P<title>[^\[]+?)\s*\((?P<doc>[^\)]+)\)\]')


def extract_citations(text: str) -> List[Tuple[str, str]]:
    """Extract citation tuples from generated text."""

    matches = CITATION_PATTERN.findall(text)
    return [(title.strip(), doc.strip()) for title, doc in matches]
