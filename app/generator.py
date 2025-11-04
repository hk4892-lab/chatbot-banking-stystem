"""SLM-based grounded generation using transformers."""
from typing import List, Dict, Literal, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SLMGenerator:
    """Small Language Model generator with grounded prompting."""

    def __init__(self, model_name: str):
        """
        Initialize SLM generator.
        
        Args:
            model_name: HuggingFace model name (e.g., microsoft/phi-3-mini-4k-instruct)
        """
        self.model_name = model_name
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if not self._loaded:
            print(f"Loading SLM: {self.model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
            )
            
            # Set pad token if not defined
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self._loaded = True

    def make_prompt(
        self,
        query: str,
        passages: List[Dict],
        lang: Literal["EN", "HI", "TA"],
    ) -> str:
        """
        Build grounded prompt with context.
        
        Args:
            query: Masked user query
            passages: Retrieved context passages
            lang: Detected language
            
        Returns:
            Formatted prompt string
        """
        # System instruction
        system = (
            "You are a helpful banking support assistant. "
            "Answer STRICTLY based on the CONTEXT provided below. "
            "If the information is not in the context, politely decline and refer to official channels. "
            "Always cite sources using format [Title (doc_id)]. "
            "Never reveal or unmask PII tokens. "
            "Keep answers concise and accurate."
        )
        
        # Build context section
        context_parts = []
        for passage in passages:
            title = passage.get("title", "Unknown")
            doc_id = passage.get("doc_id", "unknown")
            content = passage.get("content", "")
            context_parts.append(f"[{title} ({doc_id})]\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Construct full prompt
        prompt = f"""<|system|>
{system}

CONTEXT:
{context}

<|user|>
{query}

<|assistant|>
"""
        return prompt

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate response from prompt.
        
        Args:
            prompt: Full prompt with context
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        self._load_model()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and extract only new tokens
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Try to extract only the assistant response
        if "<|assistant|>" in full_text:
            response = full_text.split("<|assistant|>")[-1].strip()
        else:
            # Fallback: remove the prompt
            prompt_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            response = full_text[len(prompt_text):].strip()
        
        # Trim excessive whitespace
        response = " ".join(response.split())
        
        return response
