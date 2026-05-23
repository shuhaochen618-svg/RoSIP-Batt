"""
Rotary Position Embedding (RoPE) for 1-D time-series Transformer.

Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position
Embedding", arXiv 2104.09864 (2021).

Adapted for battery charging sequences where positional relationships between
time steps carry physical meaning (voltage ramp rate, current plateau duration).
"""

import torch
import torch.nn as nn


def precompute_rope_cache(
    d_head: int,
    max_len: int,
    base: float = 10000.0,
    device: str = "cpu",
):
    """Pre-compute cosine and sine tables for RoPE.

    Args:
        d_head: Dimension per attention head (must be even).
        max_len: Maximum sequence length (T + 1 for [CLS]).
        base: Frequency base (default 10000 following Su et al.).
        device: Target device.

    Returns:
        cos, sin: Each of shape (max_len, d_head // 2).
    """
    assert d_head % 2 == 0, f"d_head must be even, got {d_head}"
    inv_freq = 1.0 / (
        base ** (torch.arange(0, d_head, 2, device=device).float() / d_head)
    )
    t = torch.arange(max_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (max_len, d_head/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embedding to query or key tensor.

    Args:
        x: (B, h, L, d_head) — query or key.
        cos: (max_len, d_head // 2) — pre-computed cosine cache.
        sin: (max_len, d_head // 2) — pre-computed sine cache.

    Returns:
        Rotated tensor of the same shape as x.
    """
    seq_len = x.shape[-2]
    # Split even/odd dimensions
    x1 = x[..., 0::2]  # (B, h, L, d_head/2)
    x2 = x[..., 1::2]  # (B, h, L, d_head/2)
    # Broadcast cos/sin to (1, 1, L, d_head/2)
    cos_slice = cos[None, None, :seq_len, :]  # (1, 1, L, d_head/2)
    sin_slice = sin[None, None, :seq_len, :]
    # Rotate
    rotated = torch.stack(
        [x1 * cos_slice - x2 * sin_slice,
         x1 * sin_slice + x2 * cos_slice],
        dim=-1,
    )
    return rotated.flatten(-2)  # (B, h, L, d_head)


class RoPECache(nn.Module):
    """Module wrapper that holds a cached RoPE table as a buffer."""

    def __init__(self, d_head: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        cos, sin = precompute_rope_cache(d_head, max_len, base)
        # Register as buffers (not parameters) — they move with .to(device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to input tensor (B, h, L, d_head)."""
        return apply_rope(x, self.cos, self.sin)
