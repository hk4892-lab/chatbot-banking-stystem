"""Integration helpers for the optional small language model."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from .config import MODEL_ID, USE_SLM


LOGGER = logging.getLogger("bankbot.slm")


def _infer_device_kwargs(torch_module):
    if torch_module.cuda.is_available():
        return {"torch_dtype": torch_module.bfloat16, "device_map": "auto"}
    return {"torch_dtype": torch_module.float32}


@lru_cache(maxsize=1)
def _load_pipeline():
    if not USE_SLM:
        LOGGER.info("SLM usage disabled via configuration.")
        return None

    try:
        import torch  # type: ignore[import]
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            pipeline,
        )
    except Exception as exc:  # pragma: no cover - import failure path
        LOGGER.warning("SLM dependencies unavailable: %s", exc)
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        device_kwargs = _infer_device_kwargs(torch)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **device_kwargs)
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
        LOGGER.info("Loaded SLM pipeline: %s", MODEL_ID)
        return text_pipe
    except Exception as exc:  # pragma: no cover - runtime load failure
        LOGGER.error("Failed to load SLM pipeline: %s", exc)
        return None


def slm_available() -> bool:
    return _load_pipeline() is not None


def _build_prompt(system: str, messages: List[dict]) -> str:
    segments = []
    if system:
        segments.append(f"<|system|>\n{system.strip()}")
    for message in messages:
        role = message.get("role", "user")
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if role not in {"user", "assistant", "system"}:
            role = "user"
        segments.append(f"<|{role}|>\n{content}")
    segments.append("<|assistant|>\n")
    return "\n".join(segments)


def slm_generate(system: str, messages: List[dict]) -> str:
    pipeline_fn = _load_pipeline()
    if pipeline_fn is None:
        return ""

    prompt = _build_prompt(system, messages)
    try:
        outputs = pipeline_fn(
            prompt,
            eos_token_id=pipeline_fn.tokenizer.eos_token_id,  # type: ignore[attr-defined]
        )
        if not outputs:
            return ""
        generated = outputs[0].get("generated_text", "").strip()
        return generated
    except Exception as exc:  # pragma: no cover - runtime generation failure
        LOGGER.error("SLM generation failed: %s", exc)
        return ""


__all__ = ["slm_generate", "slm_available"]
