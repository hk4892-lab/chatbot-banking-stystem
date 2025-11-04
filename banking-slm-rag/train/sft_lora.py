"""LoRA fine-tuning script for the small language model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def format_prompt(entry: dict) -> str:
    instruction = entry.get('instruction', '').strip()
    input_text = entry.get('input', '').strip()
    output_text = entry.get('output', '').strip()
    if input_text:
        return f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nAnswer:\n{output_text}"
    return f"Instruction:\n{instruction}\n\nAnswer:\n{output_text}"


def build_dataset(entries: Iterable[dict]) -> Dataset:
    texts = [{'text': format_prompt(entry)} for entry in entries]
    return Dataset.from_list(texts)


def train(
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    model_name: str,
    lr: float,
    epochs: int,
    batch_size: int,
    lora_r: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading datasets from {train_path} and {val_path}')
    train_dataset = build_dataset(load_jsonl(train_path))
    val_dataset = build_dataset(load_jsonl(val_path))

    print(f'Loading base model {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        bias='none',
        task_type='CAUSAL_LM',
    )

    training_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        weight_decay=0.0,
        logging_steps=5,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        max_seq_length=512,
        warmup_ratio=0.03,
        bf16=False,
        fp16=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_config,
        peft_config=peft_config,
        dataset_text_field='text',
        packing=False,
    )

    print('Starting fine-tuning...')
    trainer.train()
    print('Training complete, saving model...')
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f'Model saved to {output_dir}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fine-tune the SLM using LoRA.')
    parser.add_argument('train_path', type=Path, help='Training dataset JSONL path.')
    parser.add_argument('val_path', type=Path, help='Validation dataset JSONL path.')
    parser.add_argument('output_dir', type=Path, help='Directory to save LoRA weights.')
    parser.add_argument('--model', default='microsoft/phi-3-mini-4k-instruct')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lora-r', type=int, default=8)
    return parser.parse_args()


if __name__ == '__main__':  # pragma: no cover
    args = parse_args()
    train(
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        model_name=args.model,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
    )
