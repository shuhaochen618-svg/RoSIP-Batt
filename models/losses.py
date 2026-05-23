"""
Homoscedastic Uncertainty-Weighted Multi-Task Loss.

Reference: Kendall, Gal & Cipolla, "Multi-Task Learning Using Uncertainty
to Weigh Losses for Scene Geometry and Semantics", CVPR 2018.

Joint loss for SOH + RUL prediction:
    L(θ, s_S, s_R) = 0.5 · exp(-s_S) · L_S + 0.5 · exp(-s_R) · L_R
                    + 0.5 · s_S + 0.5 · s_R

where:
    s_S = log(σ²_SOH)   — learnable log-variance for SOH task
    s_R = log(σ²_RUL)   — learnable log-variance for RUL task
    L_S = SOH_LOSS_SCALE · MSE(ŜOH, SOH)
    L_R = HuberLoss(R̂UL, RUL) or MSE(R̂UL, RUL)
"""

import torch
import torch.nn.functional as F
from typing import Optional


def uncertainty_weighted_loss(
    soh_hat: torch.Tensor,
    soh: torch.Tensor,
    rul_hat: torch.Tensor,
    rul: torch.Tensor,
    s_soh: torch.Tensor,
    s_rul: torch.Tensor,
    soh_loss_scale: float = 500.0,
    rul_huber_delta: float = 30.0,
    rul_huber_scale: float = 0.01,
    s_clamp: Optional[tuple] = (-5.0, 3.0),
    use_huber: bool = True,
):
    """Compute the homoscedastic uncertainty-weighted joint loss.

    Args:
        soh_hat: (B,) — predicted SOH ∈ (0, 1).
        soh: (B,) — ground truth SOH.
        rul_hat: (B,) — predicted RUL.
        rul: (B,) — ground truth RUL.
        s_soh: scalar Parameter — log-variance for SOH.
        s_rul: scalar Parameter — log-variance for RUL.
        soh_loss_scale: Scale factor for SOH MSE (default 500.0).
        rul_huber_delta: Delta threshold for Huber loss.
        rul_huber_scale: Scale factor for RUL Huber loss.
        s_clamp: Clamp range for log-variance parameters.
        use_huber: Whether to use Huber loss for RUL instead of MSE.

    Returns:
        loss: scalar — total weighted loss.
        L_soh: scalar — SOH MSE loss (scaled).
        L_rul: scalar — RUL loss (scaled).
        w_soh: scalar — effective SOH weight = 0.5 * exp(-s_soh).
        w_rul: scalar — effective RUL weight = 0.5 * exp(-s_rul).
    """
    # Clamp log-variance for stability
    if s_clamp is not None:
        s_soh_c = torch.clamp(s_soh, s_clamp[0], s_clamp[1])
        s_rul_c = torch.clamp(s_rul, s_clamp[0], s_clamp[1])
    else:
        s_soh_c = s_soh
        s_rul_c = s_rul

    # 1. SOH Loss (MSE scaled to be in the same order of magnitude as RUL loss)
    L_soh = ((soh_hat - soh) ** 2).mean() * soh_loss_scale

    # 2. RUL Loss (Huber loss is more robust to outlier cycles in early/late phases)
    if use_huber:
        L_rul = F.huber_loss(rul_hat, rul, delta=rul_huber_delta) * rul_huber_scale
    else:
        L_rul = ((rul_hat - rul) ** 2).mean()

    # 3. Uncertainty-Weighted Combination
    loss = (
        0.5 * torch.exp(-s_soh_c) * L_soh
        + 0.5 * torch.exp(-s_rul_c) * L_rul
        + 0.5 * s_soh_c
        + 0.5 * s_rul_c
    )

    # Effective weights for monitoring
    w_soh = (0.5 * torch.exp(-s_soh_c)).detach()
    w_rul = (0.5 * torch.exp(-s_rul_c)).detach()

    return loss, L_soh.detach(), L_rul.detach(), w_soh, w_rul
