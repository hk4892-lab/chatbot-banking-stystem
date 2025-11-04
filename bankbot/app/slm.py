"""Small language model helpers for the Phi-3 mini model."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, List

from .config import MODEL_ID, USE_SLM


LOGGER = logging.getLogger("bankbot.slm")


def _device_kwargs(torch_module):
    if torch_module.cuda.is_available():
        return {"torch_dtype": torch_module.bfloat16, "device_map": "auto"}
    return {"torch_dtype": torch_module.float32}


@lru_cache(maxsize=1)
def get_pipe():
    if not USE_SLM:
        return None

    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except Exception as exc:  # pragma: no cover - optional dependency failure
        LOGGER.warning("Transformers or torch unavailable: %s", exc)
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **_device_kwargs(torch))
        text_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.2,
            repetition_penalty=1.05,
            return_full_text=False,
        )
        return text_pipe
    except Exception as exc:  # pragma: no cover - runtime failure
        LOGGER.warning("Failed to load SLM '%s': %s", MODEL_ID, exc)
        return None


def slm_available() -> bool:
    return get_pipe() is not None


def phi_prompt(system: str, messages: Iterable[dict]) -> str:
    lines: List[str] = []
    system = (system or "").strip() or "You are BankBot, a concise and safe banking assistant."
    lines.append("<|system|>")
    lines.append(system)
    for message in messages:
        role = message.get("role", "user")
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if role not in {"system", "user", "assistant"}:
            role = "user"
        lines.append(f"<|{role}|>")
        lines.append(content)
    lines.append("<|assistant|>")
    lines.append("")
    return "\n".join(lines)


def generate(
    system: str,
    messages: List[dict],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> str:
    pipe = get_pipe()
    if pipe is None:
        return ""

    prompt = phi_prompt(system, messages)
    try:
        outputs = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=pipe.tokenizer.eos_token_id,  # type: ignore[attr-defined]
        )
        if not outputs:
            return ""
        return (outputs[0].get("generated_text") or "").strip()
    except Exception as exc:  # pragma: no cover - generation failure
        LOGGER.warning("SLM generation failed: %s", exc)
        return ""


__all__ = ["get_pipe", "slm_available", "phi_prompt", "generate"]

