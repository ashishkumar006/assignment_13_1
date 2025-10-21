"""
SmolLM2-135M Model Implementation
Based on LLaMA architecture with optimizations for small-scale training

Architecture Details:
- Model Size: 135M parameters
- Hidden Size: 576
- Intermediate Size: 1536
- Number of Layers: 30
- Attention Heads: 9 (3 KV heads - Grouped Query Attention)
- Vocab Size: 49152
- Max Position Embeddings: 8192
- RoPE Theta: 100000
- Activation: SiLU
- Normalization: RMSNorm (eps=1e-5)

Key Optimizations:
1. Weight Sharing (tie_word_embeddings=True): Input and output embeddings are shared
2. Residual Standard Scaling: Gradients are scaled by 1/sqrt(2*num_layers) for stability
3. Grouped Query Attention: 9 query heads share 3 key-value heads for efficiency
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 100000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate position indices
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Create rotation matrices
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    9 query heads share 3 key-value heads
    This reduces memory and computation while maintaining performance
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.num_kv_heads = config['num_key_value_heads']
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config['max_position_embeddings'],
            base=config['rope_theta']
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, seq_length)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Expand K and V for grouped query attention
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class MLP(nn.Module):
    """Feed-forward network with SiLU activation"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.intermediate_size = config['intermediate_size']
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: gate(x) * up(x)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer block with attention and MLP"""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config['hidden_size']
        
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.post_attention_layernorm = RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        
        # Residual scaling factor for gradient stability
        # Scale by 1/sqrt(2*num_layers) as mentioned in the optimization
        self.residual_scale = 1.0 / math.sqrt(2 * config['num_hidden_layers'])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual connection and scaling
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states * self.residual_scale
        
        # MLP with residual connection and scaling
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_scale
        
        return hidden_states


class SmolLM2Model(nn.Module):
    """
    SmolLM2-135M Base Model
    
    This implements the core transformer architecture with:
    - Token embeddings (shared with output layer)
    - 30 transformer blocks
    - Final layer normalization
    
    Key Features:
    - Weight sharing between input and output embeddings (reduces params by ~28M)
    - Residual standard scaling for training stability
    - Grouped query attention for efficiency
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        
        # Token embeddings (will be shared with output layer)
        self.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'])
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config['num_hidden_layers'])
        ])
        
        # Final normalization
        self.norm = RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with variance scaling"""
        std = self.config['initializer_range']
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare attention mask (causal mask for autoregressive generation)
        if attention_mask is None:
            batch_size, seq_length = input_ids.shape
            attention_mask = torch.triu(
                torch.full((seq_length, seq_length), float('-inf'), device=input_ids.device),
                diagonal=1
            )
        
        # Pass through transformer blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class SmolLM2ForCausalLM(nn.Module):
    """
    SmolLM2-135M for Causal Language Modeling
    
    Adds the language modeling head on top of the base model.
    The LM head shares weights with the input embeddings (weight tying).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = SmolLM2Model(config)
        
        # Language modeling head (output projection)
        # Note: This will share weights with embed_tokens
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        
        # Tie weights between input embeddings and output projection
        # This is a key optimization that reduces parameters
        if config['tie_word_embeddings']:
            self.lm_head.weight = self.model.embed_tokens.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with variance scaling"""
        std = self.config['initializer_range']
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        # Get hidden states from base model
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config['vocab_size']),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return loss, logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> torch.Tensor:
        """Simple greedy generation with temperature and top-k sampling"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                _, logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS token (assuming 0 is EOS)
                if next_token.item() == 0:
                    break
        
        return input_ids


def get_model_config():
    """
    Returns the SmolLM2-135M configuration
    
    This matches the official HuggingFace model configuration
    """
    return {
        'hidden_size': 576,
        'intermediate_size': 1536,
        'num_hidden_layers': 30,
        'num_attention_heads': 9,
        'num_key_value_heads': 3,
        'vocab_size': 49152,
        'max_position_embeddings': 8192,
        'rope_theta': 100000.0,
        'rms_norm_eps': 1e-5,
        'initializer_range': 0.041666666666666664,
        'tie_word_embeddings': True,
        'hidden_act': 'silu',
    }


def count_parameters(model):
    """
    Count total and trainable parameters
    
    Returns detailed breakdown of parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Detailed breakdown
    breakdown = {
        'embeddings': model.model.embed_tokens.weight.numel(),
        'transformer_blocks': sum(p.numel() for layer in model.model.layers for p in layer.parameters()),
        'final_norm': sum(p.numel() for p in model.model.norm.parameters()),
        'lm_head': 0 if model.config['tie_word_embeddings'] else model.lm_head.weight.numel(),
    }
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'breakdown': breakdown,
        'total_millions': total_params / 1e6
    }


if __name__ == "__main__":
    # Test model creation and parameter counting
    config = get_model_config()
    model = SmolLM2ForCausalLM(config)
    
    param_info = count_parameters(model)
    print("=" * 60)
    print("SmolLM2-135M Model Architecture")
    print("=" * 60)
    print(f"\nTotal Parameters: {param_info['total']:,} ({param_info['total_millions']:.2f}M)")
    print(f"Trainable Parameters: {param_info['trainable']:,}")
    print(f"\nParameter Breakdown:")
    print(f"  - Embeddings: {param_info['breakdown']['embeddings']:,}")
    print(f"  - Transformer Blocks: {param_info['breakdown']['transformer_blocks']:,}")
    print(f"  - Final Norm: {param_info['breakdown']['final_norm']:,}")
    print(f"  - LM Head: {param_info['breakdown']['lm_head']:,} (shared with embeddings)")
    print("\n" + "=" * 60)
    
    # Test forward pass
    batch_size, seq_length = 2, 128
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_length))
    labels = torch.randint(0, config['vocab_size'], (batch_size, seq_length))
    
    loss, logits = model(input_ids, labels=labels)
    print(f"\nTest Forward Pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print("=" * 60)
