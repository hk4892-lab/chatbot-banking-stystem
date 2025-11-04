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


class SLMLoadError(RuntimeError):
    """Raised when the SLM cannot be loaded."""


@lru_cache(maxsize=1)
def _load_pipeline():
    if not USE_SLM:
        LOGGER.info("SLM usage disabled via configuration.")
        return None

    try:
        import torch  # type: ignore[import]
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except Exception as exc:  # pragma: no cover - import failure path
        raise SLMLoadError(
            "SLM dependencies are missing. Install torch and transformers to enable the Phi-3 mini model."
        ) from exc

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
        raise SLMLoadError(
            f"Failed to download or load model '{MODEL_ID}'. Check network connectivity and model access permissions."
        ) from exc


def slm_available() -> bool:
    if not USE_SLM:
        return False

    try:
        return _load_pipeline() is not None
    except SLMLoadError as exc:  # pragma: no cover - handled gracefully
        LOGGER.warning("SLM unavailable: %s", exc)
        return False


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
    try:
        pipeline_fn = _load_pipeline()
    except SLMLoadError as exc:
        LOGGER.error("SLM generation aborted: %s", exc)
        return ""

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
