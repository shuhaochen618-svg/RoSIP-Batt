"""
MTL Battery Transformer — Main Model
=====================================
Multi-task Transformer for joint SOH + RUL prediction with homoscedastic
uncertainty weighting.

Data flow:
    X (B, 256, 3)
       │
       ▼ Conv1D / Linear Projection (3 → d) -> (B, 256, d)
       │
       ▼ Prepend Dual [CLS] tokens          -> (B, 258, d)
       │
       ▼ Transformer Encoder × L (Pre-LN + RoPE + MHA + FFN)
       │
       ▼ LayerNorm
       │
       ├──▶ Extract z_soh = output[:, 0, :]  -> (B, d)  [CLS_SOH representation]
       └──▶ Extract z_rul = output[:, 1, :]  -> (B, d)  [CLS_RUL representation]
       │
       ├──▶ SOH Head (Gated z_soh + detach(z_rul)) -> Sigmoid -> ŜOH ∈ (0, 1]
       │                                                         │
       │                                                         ▼ (gradient detach & normalize)
       │                                                      soh_feat
       │                                                         │
       └──▶ RUL Head (z_rul + extra_feats + soh_feat) -> Softplus -> R̂UL ≥ 0

Learnable parameters:
    - s_soh: log(σ²_SOH) — log-variance for SOH task (init: 0.0)
    - s_rul: log(σ²_RUL) — log-variance for RUL task (init: 0.0)
"""

import torch
import torch.nn as nn

from models.transformer import TransformerBlock
from models.heads import SOHHead, RULHead


class MTLBatteryTransformer(nn.Module):
    """Shared Transformer Encoder with dual task heads for SOH + RUL.

    Args:
        d_model: Embedding / hidden dimension. Default 128.
        n_layers: Number of Transformer encoder blocks. Default 2.
        n_heads: Number of attention heads. Default 4.
        seq_len: Fixed resampled sequence length. Default 256.
        input_channels: Number of input channels (V, I, T). Default 3.
        dropout: Dropout rate for attention and FFN. Default 0.1.
        s_soh_init: Initial value for log-variance s_SOH. Default 0.0.
        s_rul_init: Initial value for log-variance s_RUL. Default 0.0.
        use_rope: Whether to use Rotary Position Embedding. Default True.
        use_dual_cls: Whether to use dual CLS tokens. Default True.
        use_gate: Whether to use Gated Fusion for SOH. Default True.
        use_detach: Whether to detach gradients flowing from SOH to RUL. Default True.
        use_conv_embed: Whether to use 1D CNN Embedding. Default True (R19 default).
        use_extra_features: Whether to use external physical features. Default True (R19 default).
        use_soh_in_rul: Whether to inject predicted SOH into RUL head. Default True (R19 default).
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        seq_len: int = 256,
        input_channels: int = 3,
        dropout: float = 0.1,
        s_soh_init: float = 0.0,
        s_rul_init: float = 0.0,
        use_rope: bool = True,
        use_dual_cls: bool = True,
        use_gate: bool = True,
        use_detach: bool = True,
        use_conv_embed: bool = True,       # Defaults to True for R19
        use_extra_features: bool = True,   # Defaults to True for R19
        use_soh_in_rul: bool = True,       # Defaults to True for R19
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.use_rope = use_rope
        self.use_dual_cls = use_dual_cls
        self.use_gate = use_gate
        self.use_detach = use_detach
        self.use_conv_embed = use_conv_embed
        self.use_extra_features = use_extra_features
        self.use_soh_in_rul = use_soh_in_rul

        # ─── Input Embedding ───
        if use_conv_embed:
            self.input_proj = nn.Sequential(
                nn.Conv1d(input_channels, d_model, kernel_size=5, padding=2, padding_mode='replicate'),
                nn.BatchNorm1d(d_model),
                nn.GELU()
            )
        else:
            self.input_proj = nn.Linear(input_channels, d_model)
        
        num_cls = 2 if use_dual_cls else 1
        if not use_rope:
            self.pos_emb = nn.Parameter(torch.zeros(1, seq_len + num_cls, d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # ─── [CLS] Tokens ───
        if use_dual_cls:
            self.cls_soh = nn.Parameter(torch.zeros(1, 1, d_model))
            self.cls_rul = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_soh, std=0.02)
            nn.init.trunc_normal_(self.cls_rul, std=0.02)
            # ─── Per-Dimension Gated Fusion ───
            if use_gate:
                # gate = sigmoid(W @ z_soh + b) ∈ (0,1)^d, 初始化 b=-3 使 gate≈0.047
                # 训练初期近似纯 Dual-CLS, 模型逐步学习最优融合比例
                self.fusion_gate = nn.Linear(d_model, d_model, bias=True)
                nn.init.zeros_(self.fusion_gate.weight)
                nn.init.constant_(self.fusion_gate.bias, -3.0)
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ─── Transformer Encoder ───
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout, use_rope=use_rope)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)

        # ─── Task Heads ───
        self.soh_head = SOHHead(d_model)
        self.rul_head = RULHead(d_model, use_extra_features=use_extra_features, use_soh_in_rul=use_soh_in_rul)

        # ─── Learnable Log-Variance (Kendall 2018) ───
        self.s_soh = nn.Parameter(torch.tensor(s_soh_init))
        self.s_rul = nn.Parameter(torch.tensor(s_rul_init))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with Xavier uniform and biases to zero."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        X: torch.Tensor,
        cycle_idx: torch.Tensor = None,
        duration: torch.Tensor = None,
        return_attn: bool = False,
    ):
        """
        Args:
            X: (B, T, 3) — charging segment (V, I, T) time series.
            cycle_idx: (B,) — cycle index of each sample.
            duration: (B,) — physical CC charging duration of each sample.
            return_attn: If True, collect attention weights from all layers.

        Returns:
            soh_hat: (B,) — predicted SOH ∈ (0, 1).
            rul_hat: (B,) — predicted RUL ≥ 0.
            attn_maps: list of (B, h, L, L) if return_attn else None.
        """
        B = X.shape[0]

        # Input projection: (B, T, 3) → (B, T, d)
        if self.use_conv_embed:
            h = X.transpose(1, 2)  # (B, input_channels, T)
            h = self.input_proj(h)  # (B, d_model, T)
            h = h.transpose(1, 2)  # (B, T, d_model)
        else:
            h = self.input_proj(X)

        # Prepend [CLS]
        if self.use_dual_cls:
            cls_soh = self.cls_soh.expand(B, -1, -1)
            cls_rul = self.cls_rul.expand(B, -1, -1)
            h = torch.cat([cls_soh, cls_rul, h], dim=1)  # (B, T+2, d)
        else:
            cls = self.cls_token.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)  # (B, T+1, d)
        
        if not self.use_rope:
            h = h + self.pos_emb[:, :h.size(1), :]

        # Transformer encoder blocks
        attn_maps = [] if return_attn else None
        for blk in self.blocks:
            h, attn_w = blk(h, return_attn=return_attn)
            if return_attn and attn_w is not None:
                attn_maps.append(attn_w)

        # Final LayerNorm
        h = self.ln_out(h)

        # Extract representations
        if self.use_dual_cls:
            z_soh = h[:, 0, :]
            z_rul = h[:, 1, :]
            
            # Detach gradient path from SOH to RUL if requested
            z_rul_ref = z_rul.detach() if self.use_detach else z_rul
            
            if self.use_gate:
                # Per-Dimension Gated Fusion: gate ∈ (0,1)^d
                gate = torch.sigmoid(self.fusion_gate(z_soh))  # (B, d_model)
                z_soh_fused = z_soh + gate * z_rul_ref
            else:
                # Simple additive fusion
                z_soh_fused = z_soh + z_rul_ref
        else:
            z_soh_fused = h[:, 0, :]
            z_rul = h[:, 0, :]

        # Task predictions
        soh_hat = self.soh_head(z_soh_fused)
        soh_feat = None
        if self.use_soh_in_rul:
            # Scale and centralize predicted SOH: SOH is typically between 0.7 and 1.0.
            # Shifting and scaling maps the region of interest to [-1.0, 2.0] for stable training.
            soh_feat = (soh_hat.detach() - 0.8) / 0.1

        if (self.use_extra_features and cycle_idx is not None and duration is not None) or self.use_soh_in_rul:
            extra_feat = None
            if self.use_extra_features and cycle_idx is not None and duration is not None:
                extra_feat = torch.stack([cycle_idx, duration], dim=-1)
            rul_hat = self.rul_head(z_rul, extra_feat=extra_feat, soh_feat=soh_feat)
        else:
            rul_hat = self.rul_head(z_rul)

        if return_attn:
            extra = {"attn_maps": attn_maps}
            if self.use_dual_cls and self.use_gate:
                extra["gate"] = gate
            return soh_hat, rul_hat, extra

        return soh_hat, rul_hat, None

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_uncertainty(self):
        """Return calibrated noise standard deviations.

        Returns:
            sigma_soh: float — σ_SOH = exp(s_SOH / 2)
            sigma_rul: float — σ_RUL = exp(s_RUL / 2)
        """
        sigma_soh = torch.exp(self.s_soh / 2).item()
        sigma_rul = torch.exp(self.s_rul / 2).item()
        return sigma_soh, sigma_rul
