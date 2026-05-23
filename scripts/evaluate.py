"""
Evaluation and Inference Script for the MTL Battery Transformer (RoSIP-Batt).

Loads a saved checkpoint, runs predictions on the dataset, reports performance metrics
(SOH RMSE/MAE, RUL MAE/MAPE), and plots the results.
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Fix import paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.pkl_loader import load_dataset
from data.dataset import BatteryCycleDataset, cell_level_split
from models.mtl_model import MTLBatteryTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MTL Battery Transformer (RoSIP-Batt)")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to the saved checkpoint (.pth)"
    )
    parser.add_argument(
        "--data-root", 
        type=str, 
        default=None, 
        help="Custom path to raw pickle directory"
    )
    parser.add_argument(
        "--plot-dir", 
        type=str, 
        default=os.path.join(PROJECT_ROOT, "results", "plots"),
        help="Directory to save evaluation plots"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda / cpu)"
    )
    return parser.parse_args()


def plot_cell_predictions(cell_id, soh_true, soh_pred, rul_true, rul_pred, save_path):
    """Generate a comparison plot for a cell's SOH and RUL prediction trajectories."""
    cycles = np.arange(len(soh_true))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. SOH comparison
    ax1.plot(cycles, soh_true, 'k-', label='True SOH', linewidth=2)
    ax1.plot(cycles, soh_pred, 'r--', label='Predicted SOH (RoSIP-Batt)', linewidth=2)
    ax1.set_ylabel('State of Health (SOH)')
    ax1.set_title(f'Cell {cell_id} Joint Prediction Trajectory')
    ax1.legend(loc='lower left')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. RUL comparison
    ax2.plot(cycles, rul_true, 'k-', label='True RUL', linewidth=2)
    ax2.plot(cycles, rul_pred, 'b--', label='Predicted RUL (RoSIP-Batt)', linewidth=2)
    ax2.set_ylabel('Remaining Useful Life (RUL)')
    ax2.set_xlabel('Cycles')
    ax2.legend(loc='lower left')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [PLOT] Saved prediction plot to {save_path}")


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint file {args.checkpoint} does not exist.")
        sys.exit(1)

    print("=" * 70)
    print("      EVALUATING MTL BATTERY TRANSFORMER (RoSIP-Batt)")
    print("=" * 70)
    print(f" Checkpoint: {args.checkpoint}")
    print(f" Device:     {args.device}")
    print("-" * 70)

    # ─── Load Checkpoint ───
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    cfg_model = checkpoint["model_config"]
    stats = checkpoint["norm_stats"]
    dataset_name = checkpoint["dataset"]
    model_state = checkpoint["model_state"]

    print(f"Loaded checkpoint trained on dataset: {dataset_name}")
    print(f"Model config: d_model={cfg_model['d_model']}, n_layers={cfg_model['n_layers']}, n_heads={cfg_model['n_heads']}")

    # ─── Load Data ───
    print("Loading dataset...")
    dataset_data = load_dataset(dataset_name, data_root=args.data_root, seq_len=cfg_model["seq_len"])
    cell_ids = sorted(dataset_data.keys())
    
    if not cell_ids:
        print(f"[ERROR] No cells loaded for dataset {dataset_name}.")
        sys.exit(1)

    # Reconstruct the same splits
    train_cells, val_cells, test_cells = cell_level_split(cell_ids, ratios=(0.7, 0.1, 0.2), seed=42)
    # Handle small datasets
    if not val_cells and len(train_cells) > 1:
        val_cells = [train_cells[-1]]
        train_cells = train_cells[:-1]
        
    print(f"Splits -> Train cells: {train_cells} | Val cells: {val_cells} | Test cells: {test_cells}")

    # ─── Load Model ───
    model = MTLBatteryTransformer(
        d_model=cfg_model["d_model"],
        n_layers=cfg_model["n_layers"],
        n_heads=cfg_model["n_heads"],
        seq_len=cfg_model["seq_len"],
        input_channels=cfg_model["input_channels"],
        dropout=0.0,  # Turn off dropout for evaluation
        use_rope=cfg_model.get("use_rope", True),
        use_dual_cls=cfg_model.get("use_dual_cls", True),
        use_gate=cfg_model.get("use_gate", True),
        use_detach=cfg_model.get("use_detach", True),
        use_conv_embed=cfg_model.get("use_conv_embed", True),
        use_extra_features=cfg_model.get("use_extra_features", True),
        use_soh_in_rul=cfg_model.get("use_soh_in_rul", True)
    ).to(args.device)
    
    model.load_state_dict(model_state)
    model.eval()

    # ─── Evaluate Each Test Cell Individually ───
    os.makedirs(args.plot_dir, exist_ok=True)
    
    total_soh_se = []
    total_soh_ae = []
    total_rul_ae = []
    total_rul_ape = []

    print("\nEvaluating individual test cell trajectories:")
    print("-" * 70)

    for cell_id in test_cells:
        cell_samples = dataset_data[cell_id]
        # Keep sequential order for trajectory evaluation and plotting
        cell_ds = BatteryCycleDataset(cell_samples, stats, use_log_rul=True)
        cell_loader = DataLoader(cell_ds, batch_size=64, shuffle=False)
        
        soh_pred_list, soh_true_list = [], []
        rul_pred_list, rul_true_list = [], []
        
        with torch.no_grad():
            for batch in cell_loader:
                X = batch["X"].to(args.device)
                c_idx = batch["cycle_idx"].to(args.device) if cfg_model["use_extra_features"] else None
                dur = batch["duration"].to(args.device) if cfg_model["use_extra_features"] else None
                
                soh_hat, rul_hat, _ = model(X, cycle_idx=c_idx, duration=dur)
                
                soh_pred_list.extend(soh_hat.cpu().numpy().tolist())
                soh_true_list.extend(batch["soh"].numpy().tolist())
                rul_pred_list.extend(rul_hat.cpu().numpy().tolist())
                rul_true_list.extend(batch["rul"].numpy().tolist())
                
        sp = np.array(soh_pred_list)
        st = np.array(soh_true_list)
        rp = np.expm1(np.array(rul_pred_list)) # inverse log1p
        rt = np.expm1(np.array(rul_true_list)) # inverse log1p
        
        cell_soh_rmse = np.sqrt(np.mean((sp - st)**2)) * 100
        cell_rul_mae = np.mean(np.abs(rp - rt))
        
        print(f" Cell: {cell_id:<25} | SOH RMSE: {cell_soh_rmse:6.3f}% | RUL MAE: {cell_rul_mae:6.1f} cycles")
        
        # Accumulate metrics
        total_soh_se.extend(((sp - st) ** 2).tolist())
        total_soh_ae.extend(np.abs(sp - st).tolist())
        total_rul_ae.extend(np.abs(rp - rt).tolist())
        total_rul_ape.extend((np.abs(rp - rt) / (rt + 1.0)).tolist())
        
        # Generate plot
        plot_path = os.path.join(args.plot_dir, f"{cell_id}_prediction.png")
        plot_cell_predictions(cell_id, st, sp, rt, rp, plot_path)

    # ─── Overall Metrics ───
    final_soh_rmse = np.sqrt(np.mean(total_soh_se)) * 100
    final_soh_mae = np.mean(total_soh_ae) * 100
    final_rul_mae = np.mean(total_rul_ae)
    final_rul_mape = np.mean(total_rul_ape) * 100

    print("-" * 70)
    print("OVERALL SUMMARY STATS ON TEST SET:")
    print("-" * 70)
    print(f"  SOH RMSE:  {final_soh_rmse:.4f}%")
    print(f"  SOH MAE:   {final_soh_mae:.4f}%")
    print(f"  RUL MAE:   {final_rul_mae:.2f} cycles")
    print(f"  RUL MAPE:  {final_rul_mape:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
