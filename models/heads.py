"""
Task-specific prediction heads for dual-task battery prediction.

SOH Head:  [CLS] → Linear(d, d//2) → GELU → Linear(d//2, 1) → Sigmoid
           Output: SOH ∈ (0, 1]

RUL Head:  [CLS] → Linear(d, d//2) → GELU → Linear(d//2, 1) → Softplus
           Output: RUL ≥ 0  (remaining cycle count)

Design rationale:
    - SOH uses Sigmoid because SOH ∈ (0, 1] by definition (Q_k / Q_0)
    - RUL uses Softplus (not ReLU) because:
      (a) Softplus is smooth → better gradients near zero
      (b) RUL is strictly non-negative
      (c) Avoids dead neuron issue of ReLU at zero
    - Both heads use GELU (matching the Transformer FFN activation)
    - Small hidden dim (d//2) keeps heads lightweight (~33K params each)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SOHHead(nn.Module):
    """State of Health prediction head.

    Maps shared representation z ∈ ℝ^d to SOH ∈ (0, 1].
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, d_model) — [CLS] token representation.

        Returns:
            soh: (B,) — predicted SOH in (0, 1).
        """
        return torch.sigmoid(self.net(z)).squeeze(-1)


class RULHead(nn.Module):
    """Remaining Useful Life prediction head.

    Maps shared representation z ∈ ℝ^d to RUL ≥ 0.
    """

    def __init__(self, d_model: int, use_extra_features: bool = False, use_soh_in_rul: bool = False):
        super().__init__()
        self.use_extra_features = use_extra_features
        self.use_soh_in_rul = use_soh_in_rul
        in_dim = d_model + (2 if use_extra_features else 0) + (1 if use_soh_in_rul else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, z: torch.Tensor, extra_feat: torch.Tensor = None, soh_feat: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z: (B, d_model) — [CLS] token representation.
            extra_feat: (B, 2) — optional normalized cycle_idx and duration.
            soh_feat: (B,) — optional normalized predicted SOH value.

        Returns:
            rul: (B,) — predicted RUL ≥ 0 (cycles).
        """
        feats = [z]
        if self.use_extra_features and extra_feat is not None:
            feats.append(extra_feat)
        if self.use_soh_in_rul and soh_feat is not None:
            feats.append(soh_feat.unsqueeze(-1))
            
        if len(feats) > 1:
            z = torch.cat(feats, dim=-1)
        return F.softplus(self.net(z)).squeeze(-1)
