"""
Transformer Encoder Block with Pre-Layer Normalization and RoPE.

Architecture per block:
    x → LN → MHA(RoPE) → Residual → LN → FFN(4d, GELU) → Residual

Design choices:
    - Pre-LN (Xiong et al., 2020): more stable training, no warmup-sensitive
    - RoPE on Q,K only: relative position encoding via rotation
    - FFN expansion ratio 4x: standard Transformer practice
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rope import RoPECache, apply_rope


class RoPEMultiheadAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding.

    Unlike nn.MultiheadAttention, this module applies RoPE to Q and K
    before computing attention scores, enabling relative position awareness
    without absolute position embeddings.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # RoPE cache
        self.rope = RoPECache(self.d_head, max_len=512)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ):
        """
        Args:
            x: (B, L, d_model)
            return_attn: If True, also return attention weights for visualization.

        Returns:
            out: (B, L, d_model)
            attn_weights: (B, h, L, L) if return_attn else None
        """
        B, L, _ = x.shape

        # Fused QKV → split
        qkv = self.qkv_proj(x)  # (B, L, 3*d)
        qkv = qkv.reshape(B, L, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, h, L, d_head)
        q, k, v = qkv.unbind(0)  # Each: (B, h, L, d_head)

        # Apply RoPE to Q and K
        q = apply_rope(q, self.rope.cos, self.rope.sin)
        k = apply_rope(k, self.rope.cos, self.rope.sin)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, h, L, L)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum
        out = torch.matmul(attn_weights, v)  # (B, h, L, d_head)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)  # (B, L, d)
        out = self.out_proj(out)

        if return_attn:
            return out, attn_weights
        return out, None


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Encoder Block with RoPE attention.

    Architecture:
        x → LN₁ → MHA(RoPE) → + residual → LN₂ → FFN → + residual
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_rope: bool = True):
        super().__init__()
        self.use_rope = use_rope
        self.ln1 = nn.LayerNorm(d_model)
        if use_rope:
            self.mha = RoPEMultiheadAttention(d_model, n_heads, dropout)
        else:
            self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ):
        """
        Args:
            x: (B, L, d_model)

        Returns:
            x: (B, L, d_model)
            attn_weights: (B, h, L, L) if return_attn else None
        """
        # Pre-LN + MHA + Residual
        x_ln = self.ln1(x)
        if self.use_rope:
            attn_out, attn_weights = self.mha(x_ln, return_attn=return_attn)
        else:
            attn_out, attn_weights = self.mha(x_ln, x_ln, x_ln, need_weights=return_attn)
        
        x = x + self.dropout(attn_out)

        # Pre-LN + FFN + Residual
        x = x + self.dropout(self.ffn(self.ln2(x)))

        return x, attn_weights
