"""
Common preprocessing utilities for battery cycling data.

Functions:
    - resample_to_fixed_length: Interpolate (V, I, T) to T=256 steps
    - compute_zscore_stats: Compute per-channel mean/std from training set only
    - apply_zscore: Apply z-score normalization
    - filter_anomalies: Remove corrupted / too-short samples
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def resample_to_fixed_length(
    X: np.ndarray,
    target_len: int = 256,
) -> np.ndarray:
    """Resample a variable-length time series to fixed length via linear interpolation.

    Args:
        X: (L_orig, C) — original time series, C channels.
        target_len: Target length (default 256).

    Returns:
        X_resampled: (target_len, C) — resampled time series.
    """
    L_orig, C = X.shape
    if L_orig == target_len:
        return X

    x_orig = np.linspace(0, 1, L_orig)
    x_new = np.linspace(0, 1, target_len)
    X_resampled = np.zeros((target_len, C), dtype=np.float32)
    for c in range(C):
        X_resampled[:, c] = np.interp(x_new, x_orig, X[:, c])
    return X_resampled


def compute_zscore_stats(
    samples: List[Dict],
) -> Dict[str, np.ndarray]:
    """Compute per-channel mean and std from a list of samples.

    IMPORTANT: Must only be called on the training set to prevent data leakage.

    Args:
        samples: List of dicts, each containing 'X' of shape (T, C).

    Returns:
        stats: {'mean': (C,), 'std': (C,)} — channel-wise statistics.
    """
    all_X = np.concatenate([s['X'] for s in samples], axis=0)  # (N*T, C)
    mean = all_X.mean(axis=0).astype(np.float32)  # (C,)
    std = all_X.std(axis=0).astype(np.float32)     # (C,)
    return {'mean': mean, 'std': std}


def apply_zscore(
    X: np.ndarray,
    stats: Dict[str, np.ndarray],
    eps: float = 1e-8,
) -> np.ndarray:
    """Apply channel-wise z-score normalization.

    Args:
        X: (T, C) — input time series.
        stats: {'mean': (C,), 'std': (C,)} — from compute_zscore_stats.
        eps: Small constant to prevent division by zero.

    Returns:
        X_normalized: (T, C).
    """
    return (X - stats['mean']) / (stats['std'] + eps)


def filter_anomalies(
    X: np.ndarray,
    min_length: int = 50,
) -> bool:
    """Check if a sample is valid (not corrupted).

    Args:
        X: (L, C) — raw time series BEFORE resampling.
        min_length: Minimum acceptable length (half of typical CC segment).

    Returns:
        is_valid: True if sample passes all checks.
    """
    # Check minimum length
    if X.shape[0] < min_length:
        return False

    # Check for NaN or Inf
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        return False

    # Check for saturated channels (all-same values)
    for c in range(X.shape[1]):
        if np.std(X[:, c]) < 1e-10:
            return False

    return True


def extract_cc_segment(
    voltage: np.ndarray,
    current: np.ndarray,
    temperature: np.ndarray,
    voltage_window: Tuple[float, float],
    current_threshold: float = 0.0,
    v_max_margin: float = 0.05,
) -> Optional[np.ndarray]:
    """Extract constant-current (CC) charging segment within a voltage window.

    Args:
        voltage: (L,) — raw voltage measurements.
        current: (L,) — raw current measurements.
        temperature: (L,) — raw temperature measurements.
        voltage_window: (V_min, V_max) — voltage range to keep.
        current_threshold: Minimum current to qualify as CC (> 0 for charge).
        v_max_margin: Margin below V_max to define CC end.

    Returns:
        X: (L_cc, 3) — extracted (V, I, T) segment, or None if too short.
    """
    V_min, V_max = voltage_window

    # CC mask: charging current AND within voltage window
    cc_mask = (current > current_threshold) & (voltage >= V_min) & (voltage <= V_max)

    if cc_mask.sum() < 10:
        return None

    # Extract segment
    indices = np.where(cc_mask)[0]
    V_cc = voltage[indices]
    I_cc = current[indices]
    T_cc = temperature[indices]

    X = np.stack([V_cc, I_cc, T_cc], axis=-1).astype(np.float32)  # (L_cc, 3)
    return X
