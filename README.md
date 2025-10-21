---
title: SmolLM2-135M Shakespeare Generator
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# SmolLM2-135M Shakespeare Generator

A 135M parameter language model trained on Shakespeare's works from scratch.

## Model Details

- **Architecture**: SmolLM2-135M with Grouped Query Attention (GQA)
- **Tokenizer**: HuggingFace SmolLM2 BPE tokenizer (49,152 vocab)
- **Training**: 5,050 steps on Shakespeare corpus
- **Parameters**: ~134.5M
- **Features**: 
  - 30 transformer layers
  - 9 attention heads with 3 key-value heads (GQA)
  - Residual scaling at initialization
  - RoPE positional embeddings

## Training

The model was trained from scratch for 5,050 steps on Shakespeare's complete works using:
- Batch size: 4
- Sequence length: 512
- Learning rate: 3e-4 with warmup
- Mixed precision (bfloat16)
- PyTorch Lightning framework

## Usage

Enter a prompt (character names like "ROMEO:", "JULIET:" or famous quotes) and adjust:
- **Temperature**: 0.7-0.9 for creative text
- **Top-k**: 40-50 for good diversity
- **Max Length**: Number of tokens to generate

## Limitations

- Trained for only 5,050 steps (limited training time)
- May produce repetitive or inconsistent text
- Best results with Shakespeare-style prompts
- Uses simple generation (temperature + top-k sampling only)
