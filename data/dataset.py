"""
PyTorch Dataset, cell-level split, and stratified sampler for battery data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import Dict, List, Tuple, Optional


class BatteryCycleDataset(Dataset):
    """Unified dataset for battery cycle samples.

    Each sample is a (cell, cycle) pair containing:
        X:    (T, 3)  — resampled (V, I, T) time series
        soh:  scalar   — SOH = Q_k / Q_0
        rul:  scalar   — RUL = k_eol - k
        dataset_id: int — 0=MIT, 1=NASA, 2=Stanford, 3=HUST
        cell_id: str
        cycle_idx: int
        duration: float
    """

    def __init__(self, samples: List[Dict], normalize_stats: Optional[Dict] = None, use_log_rul: bool = False):
        self.samples = samples
        self.stats = normalize_stats  # {'mean': (3,), 'std': (3,), 'cycle_mean': float, 'cycle_std': float, 'duration_mean': float, 'duration_std': float}
        self.use_log_rul = use_log_rul

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        X = torch.tensor(s['X'], dtype=torch.float32)  # (T, 3)
        if self.stats is not None:
            mean = torch.tensor(self.stats['mean'], dtype=torch.float32)
            std = torch.tensor(self.stats['std'], dtype=torch.float32)
            X = (X - mean) / (std + 1e-8)
            
        rul_val = max(s['rul'], 0.0)
        if self.use_log_rul:
            rul_val = np.log1p(rul_val)

        cycle_idx = float(s.get('cycle_idx', 0))
        duration = float(s.get('duration', 0.0))
        if self.stats and 'cycle_mean' in self.stats:
            cycle_idx = (cycle_idx - self.stats['cycle_mean']) / (self.stats['cycle_std'] + 1e-8)
            duration = (duration - self.stats['duration_mean']) / (self.stats['duration_std'] + 1e-8)

        return {
            'X': X,
            'soh': torch.tensor(s['soh'], dtype=torch.float32),
            'rul': torch.tensor(rul_val, dtype=torch.float32),
            'dataset_id': s.get('dataset_id', 0),
            'cell_id': s.get('cell_id', ''),
            'cycle_idx': torch.tensor(cycle_idx, dtype=torch.float32),
            'duration': torch.tensor(duration, dtype=torch.float32),
        }


def cell_level_split(
    cells: List[str],
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 0,
) -> Tuple[List[str], List[str], List[str]]:
    """Split cells into train/val/test sets (cell-level, NOT cycle-level).

    WARNING: Cycle-level random split leaks degradation trajectories
    across splits, leading to over-optimistic results.

    Args:
        cells: List of unique cell identifiers.
        ratios: (train, val, test) fractions. Must sum to 1.0.
        seed: Random seed for reproducibility.

    Returns:
        train_cells, val_cells, test_cells: Lists of cell IDs.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"
    rng = np.random.default_rng(seed)
    cells = list(cells)
    rng.shuffle(cells)
    n = len(cells)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return cells[:n_train], cells[n_train:n_train + n_val], cells[n_train + n_val:]


def make_cycle_level_oversampler(
    samples: List[Dict],
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """Cycle-level oversampling within source datasets.
    
    Weights each sample inversely proportional to the total number of cycles
    in its parent cell, preventing long-life cells from dominating the gradients.
    """
    if num_samples is None:
        num_samples = len(samples)
        
    cell_cycle_counts = {}
    for s in samples:
        cid = s['cell_id']
        cell_cycle_counts[cid] = cell_cycle_counts.get(cid, 0) + 1
        
    weights = [1.0 / cell_cycle_counts[s['cell_id']] for s in samples]
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
