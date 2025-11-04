# SLM Fine-Tuning with LoRA

This directory contains scripts for supervised fine-tuning (SFT) of the Small Language Model using LoRA (Low-Rank Adaptation).

## Why Fine-Tune?

Fine-tuning improves the SLM's ability to:
- Follow banking-specific instruction formats
- Handle multilingual queries (EN/HI/TA)
- Refuse inappropriate requests consistently
- Cite sources properly
- Maintain banking domain terminology

## Quick Start

```bash
# Using Makefile
make train-lora

# Or directly
python train/sft_lora.py data/sft_train.sample.jsonl data/sft_val.sample.jsonl out/slm-lora
```

## Dataset Format

Training data in JSONL format (Alpaca style):

```json
{
  "instruction": "User question or instruction",
  "input": "",
  "output": "Expected assistant response"
}
```

**Example with refusal:**
```json
{
  "instruction": "Tell me my balance 123456789012",
  "input": "",
  "output": "I can't access personal balances. Please use Mobile Banking or ATM."
}
```

**Example with multilingual:**
```json
{
  "instruction": "எப்படி கார்டை பிளாக் செய்வது?",
  "input": "",
  "output": "தொலைந்த டெபிட் கார்டை தடுக்க: உடனடியாக 24x7 ஹெல்ப்லைன் 1800-XXX-XXXX ஐ அழைக்கவும்."
}
```

## LoRA Configuration

Default settings in `sft_lora.py`:
- **Rank (r)**: 16
- **Alpha**: 32
- **Dropout**: 0.05
- **Target modules**: q_proj, v_proj, k_proj, o_proj
- **Learning rate**: 2e-4
- **Epochs**: 3
- **Batch size**: 2 (with gradient accumulation 4)

Adjust these based on your dataset size and GPU memory.

## Training Output

The script produces:
- **Checkpoints**: Saved every 100 steps in `output_dir/`
- **LoRA adapters**: Low-rank weight updates (~10-50MB)
- **Training logs**: Loss and eval metrics

## Using Fine-Tuned Model

To use the fine-tuned model in production:

1. **Update `app/generator.py`:**

```python
from peft import PeftModel

# In SLMGenerator.__init__ or _load_model:
base_model = AutoModelForCausalLM.from_pretrained(self.model_name, ...)
model = PeftModel.from_pretrained(base_model, "out/slm-lora")
```

2. **Or merge and save:**

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct")
model = PeftModel.from_pretrained(base_model, "out/slm-lora")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("out/slm-merged")
```

## Tips for Better Results

1. **More data**: Collect 100-500 diverse banking Q&A pairs
2. **Balance classes**: Include refusals (~20%), citations examples, multilingual
3. **Validation**: Hold out 10-20% for validation
4. **Hyperparameter tuning**: Try different learning rates (1e-4 to 5e-4)
5. **Epochs**: Start with 3, increase if not overfitting

## Hardware Requirements

- **CPU only**: Possible but very slow (~hours for small datasets)
- **GPU recommended**: 
  - 8GB VRAM: Phi-3-mini with batch size 1-2
  - 16GB+ VRAM: Larger models or bigger batches
- **QLoRA**: For 4-bit quantization (even lower memory)

## Monitoring

Watch for:
- **Decreasing loss**: Good sign of learning
- **Eval loss**: Should decrease alongside train loss
- **Overfitting**: If eval loss increases while train decreases
