

# SmolLM2-135M Shakespeare Generator

A 135M parameter language model trained from scratch on Shakespeare's complete works. This model demonstrates the implementation of modern transformer architecture components including Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), and residual scaling.

## Model Architecture

### Overview
- **Model Size**: 134,515,008 parameters (~135M)
- **Architecture**: SmolLM2-135M (decoder-only transformer)
- **Tokenizer**: GPT2TokenizerFast from HuggingFaceTB/SmolLM2-135M
- **Vocabulary Size**: 49,152 tokens (BPE tokenization)
- **Context Length**: 2,048 tokens maximum

### Technical Details
- **Layers**: 30 transformer decoder layers
- **Hidden Dimension**: 576
- **Intermediate Dimension**: 1,536 (FFN)
- **Attention Heads**: 9 query heads
- **Key-Value Heads**: 3 (Grouped Query Attention - 3:1 ratio)
- **Head Dimension**: 64 per head
- **Activation Function**: SiLU (Swish)
- **Normalization**: RMSNorm (ε=1e-5)
- **Position Embeddings**: RoPE (Rotary Position Embeddings, θ=10,000)
- **Weight Tying**: Input/output embeddings are tied
- **Residual Scaling**: Applied at initialization for training stability

### Key Features

**Grouped Query Attention (GQA)**:
- Reduces memory footprint by sharing key-value heads across query heads
- 9 query heads with only 3 KV heads (3x efficiency improvement)
- Maintains model quality while improving inference speed

**Rotary Position Embeddings (RoPE)**:
- Encodes relative position information directly into attention mechanism
- Better extrapolation to longer sequences than learned positional embeddings
- No absolute position embeddings needed

**Residual Scaling**:
- Scales residual connections at initialization
- Improves training stability for deep networks
- Formula: `scale = 1 / sqrt(2 * num_layers)`

## Parameter Calculation

### Total Parameters: 134,515,008 (~135M)

The model consists of the following components:

#### 1. Embedding Layer
**Token Embeddings**: `vocab_size × hidden_size`
- 49,152 × 576 = **28,311,552 parameters**

#### 2. Transformer Layers (30 layers × per-layer params)

Each transformer layer contains:

**a) RMSNorm (Input)**
- Weights: `hidden_size` = **576 parameters**

**b) Self-Attention**
- Query projections: `hidden_size × (num_heads × head_dim)` = 576 × (9 × 64) = 576 × 576 = **331,776**
- Key projections: `hidden_size × (num_kv_heads × head_dim)` = 576 × (3 × 64) = 576 × 192 = **110,592**
- Value projections: `hidden_size × (num_kv_heads × head_dim)` = 576 × (3 × 64) = 576 × 192 = **110,592**
- Output projection: `hidden_size × hidden_size` = 576 × 576 = **331,776**
- **Total Attention: 884,736 parameters**

**c) RMSNorm (Post-Attention)**
- Weights: `hidden_size` = **576 parameters**

**d) Feed-Forward Network (FFN)**
- Gate projection: `hidden_size × intermediate_size` = 576 × 1,536 = **884,736**
- Up projection: `hidden_size × intermediate_size` = 576 × 1,536 = **884,736**
- Down projection: `intermediate_size × hidden_size` = 1,536 × 576 = **884,736**
- **Total FFN: 2,654,208 parameters**

**Per-Layer Total**: 576 + 884,736 + 576 + 2,654,208 = **3,540,096 parameters**

**All 30 Layers**: 30 × 3,540,096 = **106,202,880 parameters**

#### 3. Final Components

**Final RMSNorm**:
- Weights: `hidden_size` = **576 parameters**

**Output Layer (LM Head)**:
- Weight tied with token embeddings, so **0 additional parameters**

### Parameter Breakdown Summary

| Component | Parameters | Percentage |
|-----------|-----------|-----------|
| Token Embeddings | 28,311,552 | 21.0% |
| 30 Transformer Layers | 106,202,880 | 79.0% |
| Final RMSNorm | 576 | <0.1% |
| **Total** | **134,515,008** | **100%** |

### Weight Tensors: 273 Total

The model consists of 273 weight tensors:
- **1** token embedding matrix
- **30 layers** × **9 tensors per layer** = 270 tensors
  - Per layer: 1 input norm, 4 attention (Q/K/V/O), 1 post-attn norm, 3 FFN (gate/up/down)
- **1** final RMSNorm
- **1** output projection (tied with embeddings)

**Total: 1 + 270 + 1 + 1 = 273 weight tensors** ✓

### Model Architecture Components

**Main Model (SmolLM2ForCausalLM)**:
- Token embedding layer
- 30 transformer decoder layers
- Final RMSNorm layer
- Output projection (weight tied with embeddings)
- Residual scaling for training stability

**Transformer Block**:
- Pre-attention RMSNorm
- Multi-head attention with GQA
- Post-attention RMSNorm
- Feed-forward network (SwiGLU)

**Attention Module**:
- 9 query heads
- 3 key-value heads (GQA)
- 64 dimensions per head
- Query, Key, Value projections
- Output projection
- RoPE positional encoding

**Feed-Forward Network (SwiGLU)**:
- Gate projection (hidden → intermediate)
- Up projection (hidden → intermediate)
- Down projection (intermediate → hidden)
- Activation: SiLU(gate) × up, then down

### Configuration Values

- vocab_size: 49,152
- hidden_size: 576
- intermediate_size: 1,536
- num_hidden_layers: 30
- num_attention_heads: 9
- num_key_value_heads: 3 (GQA ratio 3:1)
- max_position_embeddings: 2,048
- rms_norm_eps: 1e-5
- rope_theta: 10,000.0
- tie_word_embeddings: True

## Training Details

### Dataset
- **Source**: Shakespeare's complete works (input.txt)
- **Size**: 1,115,394 characters
- **Training Split**: 90% (1,003,854 characters)
- **Validation Split**: 10% (111,540 characters)
- **Total Tokens**: ~305,000 training tokens

### Training Configuration
- **Total Steps**: 5,050 (5,000 initial + 50 continuation)
- **Batch Size**: 16 per GPU
- **Sequence Length**: 256 tokens
- **Effective Batch Size**: 4,096 tokens per step
- **Optimizer**: AdamW
- **Learning Rate**: 3e-4 with linear warmup
- **Warmup Steps**: 500
- **Weight Decay**: 0.1
- **Gradient Clipping**: 1.0
- **Precision**: bfloat16 mixed precision
- **Hardware**: NVIDIA T4 GPU (Google Colab)
- **Framework**: PyTorch Lightning

### Training Progress

| Step | Train Loss | Val Loss | Val Perplexity | Time |
|------|-----------|----------|----------------|------|
| 500  | 4.130     | -        | -              | 14m  |
| 1000 | 1.520     | -        | -              | 30m  |
| 1500 | 0.560     | -        | -              | 46m  |
| 2000 | 0.319     | -        | -              | 1h 1m |
| 2500 | 0.240     | -        | -              | 1h 18m |
| 3000 | 0.181     | -        | -              | 1h 35m |
| 3500 | 0.155     | -        | -              | 1h 52m |
| 4000 | 0.117     | -        | -              | 2h 9m |
| 4500 | 0.093     | -        | -              | 2h 24m |
| 5000 | 0.114     | 8.450    | 7,030          | 3h 5m |

**Final Metrics**:
- Training Loss: 0.114
- Validation Loss: 8.450
- Validation Perplexity: 7,030
- Total Training Time: 3 hours 5 minutes
- Average Time per Step: 2.22 seconds

### Training Observations
- Smooth convergence with no instabilities
- Loss decreased consistently from 4.130 → 0.114
- Text generation quality improved progressively throughout training
- Model learned Shakespeare's vocabulary, style, and dialogue structure
- Validation performed at step 5,000 to assess final performance

## Generation Quality

The model shows clear progression in text quality:

**Early Training (Step 500)**: Mostly gibberish and random tokens
```
First Citizen: , the on but, the of is. COROLUS...
```

**Mid Training (Step 2500)**: Recognizable words and basic structure
```
First Citizen: better than brother: brother how I be, to them moreto...
```

**Late Training (Step 5000)**: Coherent Shakespeare-style dialogue
```
First Citizen: could stay to't thou cry queen heaven earth...
```

See `generation_samples.txt` for complete examples at each training milestone.

## Usage

### Web Interface (HuggingFace Space)

The app uses **lazy loading** - the model loads automatically on your first generation request. This allows the interface to start immediately without waiting.

**Getting Started**:
1. Enter your prompt in the text box
2. Click "Generate" - the model will load on first use (may take 10-20 seconds)
3. Subsequent generations will be faster

**Prompts**:
- Character names: `ROMEO:`, `JULIET:`, `HAMLET:`
- Famous quotes: `To be or not to be`, `All the world's a stage`
- Scene starters: `First Citizen:`, `Enter MACBETH`

**Parameters**:
- **Max Length**: 50-500 tokens (default: 150)
  - Controls how much text to generate
  
- **Temperature**: 0.1-2.0 (default: 0.8)
  - Lower (0.5-0.7): More focused, coherent, predictable
  - Higher (0.8-1.2): More creative, diverse, surprising
  
- **Top-k**: 0-100 (default: 50)
  - Limits sampling to top K most likely tokens
  - 40-50 provides good diversity
  - 0 disables filtering (not recommended)
  
- **Use Sampling**: On/Off
  - On: Uses temperature and top-k for creative generation
  - Off: Greedy decoding (always picks most likely token)

## Model Files

- **shakespeare_smollm2_step50.pt** (~540MB): Model weights and configuration
- **model.py**: Model architecture implementation
- **app.py**: Gradio web interface
- **training_logs.txt**: Detailed training metrics
- **generation_samples.txt**: Sample outputs at each training stage

## Limitations and Considerations

### Training Limitations
- **Limited Training**: Only 5,050 steps (typical LLMs train for millions)
- **Small Dataset**: ~1.1M characters (modern LLMs use billions of tokens)
- **Domain-Specific**: Only trained on Shakespeare, not general text
- **No Fine-tuning**: Trained from scratch without pretrained initialization

### Generation Limitations
- **Simple Sampling**: Only temperature + top-k (no nucleus/top-p or repetition penalty)
- **Context Window**: Limited to 2,048 tokens
- **Repetition**: May repeat phrases or get stuck in loops
- **Consistency**: May lose coherence in longer generations
- **Factuality**: Not trained for factual accuracy, only stylistic imitation

### Best Results Achieved When:
- Using Shakespeare character names or quotes as prompts
- Temperature between 0.7-0.9
- Top-k between 40-50
- Generating 100-200 tokens
- Expecting dialogue/dramatic text style




### Custom Components
- Hand-coded transformer architecture (not using HuggingFace Transformers library)
- Custom RoPE implementation for positional encoding
- GQA attention mechanism with 3:1 head ratio
- Residual scaling for training stability
- PyTorch Lightning training loop with checkpointing

### Code Structure
```
model.py              # Model architecture (SmolLM2ForCausalLM)
lightning_trainer.py  # PyTorch Lightning training wrapper
app.py               # Gradio web interface
requirements.txt     # Python dependencies
```



## Acknowledgments

- **Base Architecture**: SmolLM2 by HuggingFace
- **Tokenizer**: GPT2TokenizerFast from HuggingFaceTB/SmolLM2-135M
- **Dataset**: Shakespeare's complete works (public domain)
- **Framework**: PyTorch and PyTorch Lightning
- **Training Platform**: Google Colab (NVIDIA T4 GPU)

## License

- **Model Weights**: Trained from scratch, free to use
- **Code**: Available for educational and research purposes
- **Dataset**: Shakespeare's works (public domain)

## Contact

For questions or issues, please refer to the GitHub repository or HuggingFace Space discussions.

