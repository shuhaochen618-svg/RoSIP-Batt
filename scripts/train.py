"""
Main Training Entrypoint for the MTL Battery Transformer (R19).

Loads configurations, imports data, runs the joint SOH+RUL training loop
using Homoscedastic Uncertainty-Weighted Multi-Task Loss, and saves the best model.
"""
import os
import sys
import argparse
import yaml
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Fix import paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.pkl_loader import load_dataset, DATASET_CONFIGS
from data.dataset import BatteryCycleDataset, cell_level_split, make_cycle_level_oversampler
from models.mtl_model import MTLBatteryTransformer
from models.losses import uncertainty_weighted_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train MTL Battery Transformer (R19)")
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.path.join(PROJECT_ROOT, "configs", "default.yaml"),
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="NASA", 
        choices=["NASA", "CALCE", "Stanford", "HUST"],
        help="Target dataset to train on"
    )
    parser.add_argument(
        "--data-root", 
        type=str, 
        default=None, 
        help="Custom path to raw pickle directory"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda / cpu)"
    )
    return parser.parse_args()


def compute_norm_stats(samples):
    """Compute channel-wise mean and std, as well as cycle and duration stats."""
    all_X = np.stack([s["X"] for s in samples], axis=0)  # (N, T, 3)
    mean = torch.tensor(all_X.mean(axis=(0, 1)), dtype=torch.float32)
    std = torch.tensor(all_X.std(axis=(0, 1)), dtype=torch.float32)
    
    all_cycles = np.array([s["cycle_idx"] for s in samples], dtype=np.float32)
    all_durations = np.array([s.get("duration", 0.0) for s in samples], dtype=np.float32)
    
    return {
        "mean": mean,
        "std": std,
        "cycle_mean": float(all_cycles.mean()),
        "cycle_std": float(all_cycles.std()),
        "duration_mean": float(all_durations.mean()),
        "duration_std": float(all_durations.std()),
    }


def main():
    args = parse_args()
    
    # ─── Load Config ───
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    cfg_model = config["model"]
    cfg_train = config["training"]
    
    # Set random seed
    seed = config["data"].get("split_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 70)
    print("        TRAINING MTL BATTERY TRANSFORMER (R19)")
    print("=" * 70)
    print(f" Dataset:  {args.dataset}")
    print(f" Device:   {args.device}")
    print(f" Config:   {args.config}")
    print("-" * 70)

    # ─── Load Data ───
    print("Loading data...")
    dataset_data = load_dataset(args.dataset, data_root=args.data_root, seq_len=cfg_model["seq_len"])
    cell_ids = sorted(dataset_data.keys())
    
    if not cell_ids:
        print(f"\n[ERROR] No cells loaded for dataset {args.dataset}.")
        print("Please check your --data-root path or place .pkl files in the default data folder.")
        sys.exit(1)

    print(f"Loaded {len(cell_ids)} cells: {cell_ids}")

    # ─── Split Dataset ───
    # Perform strict cell-level splitting (no cycle-level leaks!)
    train_r, val_r, test_r = config["data"]["split_ratios"]
    train_cells, val_cells, test_cells = cell_level_split(cell_ids, ratios=(train_r, val_r, test_r), seed=seed)
    
    # Handle small datasets (e.g. NASA with only 4 cells)
    if not val_cells and len(train_cells) > 1:
        val_cells = [train_cells[-1]]
        train_cells = train_cells[:-1]
        
    print(f"Splits -> Train cells: {train_cells} | Val cells: {val_cells} | Test cells: {test_cells}")

    train_samples = [s for c in train_cells for s in dataset_data[c]]
    val_samples = [s for c in val_cells for s in dataset_data[c]]
    test_samples = [s for c in test_cells for s in dataset_data[c]]
    
    print(f"Total samples -> Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    # Compute normalization stats from training set only
    stats = compute_norm_stats(train_samples)
    print(f"Normalization Mean: {stats['mean'].numpy()} | Std: {stats['std'].numpy()}")

    # ─── Prepare Data Loaders ───
    # RUL is log-transformed during training for log-space target compression
    use_log_rul = True
    train_ds = BatteryCycleDataset(train_samples, stats, use_log_rul=use_log_rul)
    val_ds = BatteryCycleDataset(val_samples, stats, use_log_rul=use_log_rul) if val_samples else None
    test_ds = BatteryCycleDataset(test_samples, stats, use_log_rul=use_log_rul)

    bs = cfg_train["batch_size"]
    # Oversampler balances cell lengths
    sampler = make_cycle_level_oversampler(train_samples)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, pin_memory=True, drop_last=len(train_ds) > bs)
    
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=True) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, pin_memory=True)

    # ─── Build R19 Model ───
    model = MTLBatteryTransformer(
        d_model=cfg_model["d_model"],
        n_layers=cfg_model["n_layers"],
        n_heads=cfg_model["n_heads"],
        seq_len=cfg_model["seq_len"],
        input_channels=cfg_model["input_channels"],
        dropout=cfg_model["dropout"],
        s_soh_init=cfg_train["s_soh_init"],
        s_rul_init=cfg_train["s_rul_init"],
        use_rope=cfg_model["use_rope"],
        use_dual_cls=cfg_model["use_dual_cls"],
        use_gate=cfg_model["use_gate"],
        use_detach=cfg_model["use_detach"],
        use_conv_embed=cfg_model["use_conv_embed"],
        use_extra_features=cfg_model["use_extra_features"],
        use_soh_in_rul=cfg_model["use_soh_in_rul"]
    ).to(args.device)

    print(f"MTL Battery Transformer Model Size: {model.count_parameters():,} trainable parameters.")

    # ─── Optimizer & Scheduler ───
    # Separate learning rate for uncertainty log-variance (typically lr * 0.1)
    unc_params = [model.s_soh, model.s_rul]
    other_params = [p for n, p in model.named_parameters() if n not in ("s_soh", "s_rul") and p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": cfg_train["lr"], "weight_decay": cfg_train["weight_decay"]},
        {"params": unc_params, "lr": cfg_train["lr"] * 0.1, "weight_decay": 0.0},
    ])
    
    warmup_epochs = cfg_train["warmup_epochs"]
    total_epochs = cfg_train["total_epochs"]
    
    # Cosine Annealing with Warmup scheduler
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ─── Training Loop ───
    best_val_loss = float("inf")
    best_state = None
    patience_cnt = 0
    ema_val = None
    
    save_dir = os.path.join(PROJECT_ROOT, config["logging"]["save_dir"])
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "─" * 80)
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
          f"{'SOH RMSE%':>9} | {'RUL MAPE%':>9} | "
          f"{'s_soh':>6} | {'s_rul':>6} | {'w_soh':>6} | {'w_rul':>6}")
    print("─" * 80)

    for epoch in range(1, total_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            X = batch["X"].to(args.device)
            soh = batch["soh"].to(args.device)
            rul = batch["rul"].to(args.device)
            c_idx = batch["cycle_idx"].to(args.device) if cfg_model["use_extra_features"] else None
            dur = batch["duration"].to(args.device) if cfg_model["use_extra_features"] else None
            
            soh_hat, rul_hat, _ = model(X, cycle_idx=c_idx, duration=dur)
            
            loss, _, _, _, _ = uncertainty_weighted_loss(
                soh_hat, soh, rul_hat, rul, model.s_soh, model.s_rul,
                soh_loss_scale=cfg_train["soh_loss_scale"],
                rul_huber_delta=cfg_train["rul_huber_delta"],
                rul_huber_scale=cfg_train["rul_huber_scale"],
                s_clamp=cfg_train["s_clamp"]
            )
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg_train["grad_clip"])
            optimizer.step()
            train_losses.append(loss.item())
            
        scheduler.step()
        avg_train_loss = np.mean(train_losses)
        
        # ─── Validation ───
        eval_loader = val_loader if val_loader else test_loader
        model.eval()
        val_losses = []
        all_soh_pred, all_soh_true = [], []
        all_rul_pred, all_rul_true = [], []
        
        with torch.no_grad():
            for batch in eval_loader:
                X = batch["X"].to(args.device)
                soh = batch["soh"].to(args.device)
                rul = batch["rul"].to(args.device)
                c_idx = batch["cycle_idx"].to(args.device) if cfg_model["use_extra_features"] else None
                dur = batch["duration"].to(args.device) if cfg_model["use_extra_features"] else None
                
                soh_hat, rul_hat, _ = model(X, cycle_idx=c_idx, duration=dur)
                
                vl, _, _, _, _ = uncertainty_weighted_loss(
                    soh_hat, soh, rul_hat, rul, model.s_soh, model.s_rul,
                    soh_loss_scale=cfg_train["soh_loss_scale"],
                    rul_huber_delta=cfg_train["rul_huber_delta"],
                    rul_huber_scale=cfg_train["rul_huber_scale"],
                    s_clamp=cfg_train["s_clamp"]
                )
                val_losses.append(vl.item())
                
                all_soh_pred.extend(soh_hat.cpu().numpy().tolist())
                all_soh_true.extend(soh.cpu().numpy().tolist())
                all_rul_pred.extend(rul_hat.cpu().numpy().tolist())
                all_rul_true.extend(rul.cpu().numpy().tolist())

        avg_val_loss = np.mean(val_losses)
        
        # Convert SOH and RUL back to raw scale for metric logging
        sp, st = np.array(all_soh_pred), np.array(all_soh_true)
        rp, rt = np.array(all_rul_pred), np.array(all_rul_true)
        if use_log_rul:
            rp = np.expm1(rp)
            rt = np.expm1(rt)
            
        soh_rmse = np.sqrt(np.mean((sp - st)**2)) * 100
        rul_mape = np.mean(np.abs(rp - rt) / (rt + 1.0)) * 100

        s_soh_val = model.s_soh.item()
        s_rul_val = model.s_rul.item()
        w_soh_val = 0.5 * np.exp(-s_soh_val)
        w_rul_val = 0.5 * np.exp(-s_rul_val)

        # Print progress report periodically
        if epoch % 10 == 0 or epoch == 1 or epoch == total_epochs:
            print(f"  {epoch:5d} | {avg_train_loss:10.4f} | {avg_val_loss:10.4f} | "
                  f"{soh_rmse:9.3f} | {rul_mape:9.2f} | "
                  f"{s_soh_val:6.3f} | {s_rul_val:6.3f} | "
                  f"{w_soh_val:6.3f} | {w_rul_val:6.3f}")

        # Early stopping logic using Exponential Moving Average (EMA) of val loss
        if ema_val is None:
            ema_val = avg_val_loss
        else:
            ema_val = cfg_train["ema_alpha"] * ema_val + (1 - cfg_train["ema_alpha"]) * avg_val_loss

        if ema_val < best_val_loss:
            best_val_loss = ema_val
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
            # Save best checkpoint
            ckpt_path = os.path.join(save_dir, f"best_model_{args.dataset}.pth")
            torch.save({
                "model_state": best_state,
                "norm_stats": stats,
                "dataset": args.dataset,
                "r19_config": cfg_model,
            }, ckpt_path)
        else:
            if epoch >= cfg_train["min_epochs"]:
                patience_cnt += 1

        if patience_cnt >= cfg_train["patience"]:
            print(f"\nEarly stopping at epoch {epoch}!")
            break

    # ─── Final Test Set Evaluation ───
    print("\n" + "─" * 80)
    print("                 FINAL EVALUATION ON TEST SET")
    print("─" * 80)
    
    ckpt_path = os.path.join(save_dir, f"best_model_{args.dataset}.pth")
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    test_soh_pred, test_soh_true = [], []
    test_rul_pred, test_rul_true = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            X = batch["X"].to(args.device)
            soh = batch["soh"].to(args.device)
            rul = batch["rul"].to(args.device)
            c_idx = batch["cycle_idx"].to(args.device) if cfg_model["use_extra_features"] else None
            dur = batch["duration"].to(args.device) if cfg_model["use_extra_features"] else None
            
            soh_hat, rul_hat, _ = model(X, cycle_idx=c_idx, duration=dur)
            test_soh_pred.extend(soh_hat.cpu().numpy().tolist())
            test_soh_true.extend(soh.cpu().numpy().tolist())
            test_rul_pred.extend(rul_hat.cpu().numpy().tolist())
            test_rul_true.extend(rul.cpu().numpy().tolist())
            
    test_sp, test_st = np.array(test_soh_pred), np.array(test_soh_true)
    test_rp, test_rt = np.array(test_rul_pred), np.array(test_rul_true)
    if use_log_rul:
        test_rp = np.expm1(test_rp)
        test_rt = np.expm1(test_rt)
        
    test_soh_rmse = np.sqrt(np.mean((test_sp - test_st)**2)) * 100
    test_soh_mae = np.mean(np.abs(test_sp - test_st)) * 100
    test_rul_mae = np.mean(np.abs(test_rp - test_rt))
    test_rul_mape = np.mean(np.abs(test_rp - test_rt) / (test_rt + 1.0)) * 100

    print(f"  Test Cells:  {test_cells}")
    print(f"  SOH RMSE:    {test_soh_rmse:.4f}%")
    print(f"  SOH MAE:     {test_soh_mae:.4f}%")
    print(f"  RUL MAE:     {test_rul_mae:.2f} cycles")
    print(f"  RUL MAPE:    {test_rul_mape:.2f}%")
    print(f"  Checkpoint:  {ckpt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
