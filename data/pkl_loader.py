"""
Unified .pkl Data Loader for the MTL Battery Transformer.

Reads the standardised BatteryML-format .pkl files from a dataset root directory.
Supports:
  - Missing temperature channel (CALCE / Stanford) → zero-filled
  - Variable-length voltage series → resampled to fixed T=256
  - SOH & RUL label computation with configurable EOL threshold
  - Cell-level train/val/test splitting
"""
import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional

# Default data root inside the repository
_DEFAULT_DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
)

DATASET_CONFIGS = {
    "NASA": {
        "dir_name": "NASA",
        "cells": ["NASA_B0005.pkl", "NASA_B0006.pkl", "NASA_B0007.pkl", "NASA_B0018.pkl"],
        "dataset_id": 0,
        "eol_threshold": 0.7,   # NASA convention: 30% capacity fade
    },
    "CALCE": {
        "dir_name": "CALCE",
        "cells": ["CALCE_CS2_35.pkl", "CALCE_CS2_36.pkl", "CALCE_CS2_37.pkl", "CALCE_CS2_38.pkl"],
        "dataset_id": 1,
        "eol_threshold": 0.8,   # CALCE convention: 20% capacity fade
    },
    "Stanford": {
        "dir_name": "Stanford",
        "cells": None,          # Use all .pkl files in the folder
        "dataset_id": 2,
        "eol_threshold": 0.8,
    },
    "HUST": {
        "dir_name": "HUST",
        "cells": None,          # Use all .pkl files in the folder
        "dataset_id": 3,
        "eol_threshold": 0.8,
    },
}

SEQ_LEN = 256   # Fixed resampled sequence length


def resample_series(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a 1D array to target_len via linear interpolation."""
    if len(arr) == 0:
        return np.zeros(target_len)
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, arr)


def extract_cycle_features(cycle: dict, seq_len: int = SEQ_LEN) -> Optional[np.ndarray]:
    """Extract (V, I, T) features from a single cycle and resample to (seq_len, 3).

    If temperature is missing/empty/all-NaN, fills with zeros.
    Returns None if voltage or current data is too short (<10 points).
    """
    V = np.array(cycle["voltage_in_V"], dtype=np.float32)
    I = np.array(cycle["current_in_A"], dtype=np.float32)

    if len(V) < 10 or len(I) < 10:
        return None

    # Temperature: handle missing gracefully
    T_raw = cycle.get("temperature_in_C")
    if T_raw is None or len(T_raw) == 0:
        T = np.zeros(len(V), dtype=np.float32)
    else:
        T = np.array(T_raw, dtype=np.float32)
        # Replace NaN with 0
        T = np.nan_to_num(T, nan=0.0)
        if np.all(T == 0):
            T = np.zeros(len(V), dtype=np.float32)

    # Truncate to common length
    min_len = min(len(V), len(I), len(T))
    V, I, T = V[:min_len], I[:min_len], T[:min_len]

    # Resample
    V_r = resample_series(V, seq_len)
    I_r = resample_series(I, seq_len)
    T_r = resample_series(T, seq_len)

    X = np.stack([V_r, I_r, T_r], axis=-1)   # (seq_len, 3)
    return X


def load_single_cell(
    filepath: str, 
    dataset_id: int,
    eol_threshold: float = 0.8,
    seq_len: int = SEQ_LEN
) -> List[Dict]:
    """Load a single .pkl cell file and return list of cycle samples.

    Returns:
        List of dicts, each containing:
            X:          (seq_len, 3)  float32
            soh:        float
            rul:        float (cycles until EOL; -1 if never reached)
            dataset_id: int
            cell_id:    str
            cycle_idx:  int
            duration:   float
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    cell_id = os.path.splitext(os.path.basename(filepath))[0]
    cd = data["cycle_data"]
    n_cycles = len(cd)

    # Compute discharge capacity per cycle
    caps = []
    for c in cd:
        dc = c["discharge_capacity_in_Ah"]
        if dc and len(dc) > 0:
            caps.append(float(dc[-1]))
        else:
            caps.append(np.nan)
    caps = np.array(caps)

    # Nominal capacity = mean of first 5 valid cycles
    valid_early = caps[:5][~np.isnan(caps[:5])]
    if len(valid_early) == 0:
        print(f"  [SKIP] {cell_id}: no valid early capacity data.")
        return []
    q0 = float(np.mean(valid_early))

    # SOH
    soh_all = caps / q0

    # Find EOL cycle
    eol_indices = np.where(soh_all < eol_threshold)[0]
    k_eol = int(eol_indices[0]) if len(eol_indices) > 0 else -1

    samples = []
    for k in range(n_cycles):
        if np.isnan(soh_all[k]):
            continue

        soh_k = float(soh_all[k])

        # RUL computation
        if k_eol > 0:
            rul_k = float(k_eol - k)
            if rul_k < 0:
                continue  # Past EOL, skip
        else:
            # Right-censored: cell never reached EOL within observation window.
            # Use (N_total - k) as a conservative estimate.
            rul_k = float(n_cycles - 1 - k)

        # Extract features
        X = extract_cycle_features(cd[k], seq_len)
        if X is None:
            continue

        # Get CC charging duration
        t_arr = cd[k].get("time_in_s")
        if t_arr is not None and len(t_arr) > 0:
            duration = float(t_arr[-1] - t_arr[0])
        else:
            duration = 0.0

        samples.append({
            "X": X,
            "soh": soh_k,
            "rul": rul_k,
            "dataset_id": dataset_id,
            "cell_id": cell_id,
            "cycle_idx": k,
            "duration": duration,
        })

    return samples


def load_dataset(
    name: str, 
    data_root: Optional[str] = None, 
    seq_len: int = SEQ_LEN
) -> Dict[str, List[Dict]]:
    """Load all cells for a given dataset name.

    Args:
        name: One of 'NASA', 'CALCE', 'Stanford', 'HUST'.
        data_root: Path containing dataset subfolders.
        seq_len: Resampling sequence length.

    Returns:
        Dict mapping cell_id → list of cycle samples.
    """
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset name: {name}")

    config = DATASET_CONFIGS[name]
    root = data_root or _DEFAULT_DATA_ROOT
    dataset_dir = os.path.join(root, config["dir_name"])
    dataset_id = config["dataset_id"]
    eol = config["eol_threshold"]

    if not os.path.exists(dataset_dir):
        print(f"  [WARN] Dataset directory {dataset_dir} does not exist.")
        return {}

    if config["cells"] is not None:
        files = [os.path.join(dataset_dir, f) for f in config["cells"]]
    else:
        files = sorted([
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith(".pkl")
        ])

    cells_data = {}
    for fpath in files:
        if not os.path.exists(fpath):
            print(f"  [WARN] {fpath} not found, skipping.")
            continue
        samples = load_single_cell(fpath, dataset_id, eol, seq_len)
        if samples:
            cid = samples[0]["cell_id"]
            cells_data[cid] = samples
            print(f"  Loaded {cid}: {len(samples)} cycles, "
                  f"SOH [{samples[0]['soh']:.3f} → {samples[-1]['soh']:.3f}]")

    return cells_data
