#!/usr/bin/env python3
"""LoRA fine-tuning script for SLM using TRL and PEFT."""
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer


def load_sft_data(filepath: str) -> List[Dict]:
    """Load SFT dataset from JSONL."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_prompt(example: Dict) -> str:
    """Format example into prompt template."""
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]
    
    if input_text:
        prompt = f"""<|system|>
You are a helpful banking support assistant. Answer questions accurately based on your knowledge.

<|user|>
{instruction}
Context: {input_text}

<|assistant|>
{output}"""
    else:
        prompt = f"""<|system|>
You are a helpful banking support assistant. Answer questions accurately based on your knowledge.

<|user|>
{instruction}

<|assistant|>
{output}"""
    
    return prompt


def main():
    """Main training function."""
    if len(sys.argv) < 4:
        print("Usage: python sft_lora.py <train.jsonl> <val.jsonl> <output_dir>")
        print("Example: python sft_lora.py data/sft_train.sample.jsonl data/sft_val.sample.jsonl out/slm-lora")
        sys.exit(1)
    
    train_path = sys.argv[1]
    val_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Configuration
    model_name = "microsoft/phi-3-mini-4k-instruct"  # or from env
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")
    print(f"Output directory: {output_dir}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Adjust for your model
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    train_data = load_sft_data(train_path)
    val_data = load_sft_data(val_path)
    
    # Format as prompts
    train_prompts = [format_prompt(ex) for ex in train_data]
    val_prompts = [format_prompt(ex) for ex in val_data]
    
    train_dataset = Dataset.from_dict({"text": train_prompts})
    val_dataset = Dataset.from_dict({"text": val_prompts})
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        fp16=device == "cuda",
        report_to="none",  # Disable wandb/tensorboard
        load_best_model_at_end=True,
    )
    
    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=1024,
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Save
    print(f"\n💾 Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n✅ Training complete!")
    print(f"Load fine-tuned model with: model = PeftModel.from_pretrained(base_model, '{output_dir}')")


if __name__ == "__main__":
    main()
