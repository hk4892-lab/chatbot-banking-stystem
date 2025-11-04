## LoRA Fine-Tuning Guide

This directory contains `sft_lora.py`, a lightweight TRL + PEFT pipeline to adapt the base SLM on supervised instruction data.

### Quickstart

```bash
python train/sft_lora.py data/sft_train.sample.jsonl data/sft_val.sample.jsonl out/slm-lora \
  --model microsoft/phi-3-mini-4k-instruct --epochs 1 --batch-size 2 --lr 2e-4
```

The script exports LoRA adapters and tokenizer files to the specified output directory. Update `.env` to point `SLM_MODEL` to the merged weights or load the adapters via PEFT in production deployment.

### Notes

- Training configuration is CPU-friendly by default; enable CUDA by setting `device_preference=cuda` in `.env` and running on a GPU host.
- Extend the JSONL datasets with multilingual instructions and refusal demonstrations for better guardrails.
- Regularly evaluate checkpoints using `eval/evaluate.py` to track grounding quality.
