"""Train GeoProspectNet — single split, LOF fold, or full LOF-CV sweep.

CLI:
    python -m src.training.train --config configs/default.yaml --quick      # dev: 5 folds
    python -m src.training.train --config configs/default.yaml --all_folds  # full LOF-CV
    python -m src.training.train --config configs/default.yaml --fold 7     # one fold
    python -m src.training.train --config configs/default.yaml --random     # 70/15/15 split

Spatial smoothing
-----------------
During the per-batch training step we don't see every cell's neighbours, so
training forwards run with smoothing disabled. After each epoch we build a
z_lookup table of fused embeddings for all labeled cells, then re-score the
validation fold with neighbour gathering enabled. The best checkpoint is
selected on this *smoothed* validation AUROC, so the metric the model is
chosen on matches the inference-time behaviour (where smoothing is always
applied via discovery.py).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data.dataset import GeoProspectDataset, make_loader
from src.models.geoprospectnet import GeoProspectNet
from src.training.losses import CombinedLoss, FocalLoss
from src.training.utils import device_auto, load_config, save_checkpoint, seed_everything, write_json

ROOT = Path(__file__).resolve().parents[2]


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, k: int = 50) -> Dict:
    """Compute the headline metrics for one fold / split."""
    if len(set(y_true.tolist())) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan"),
                "precision_at_k": float("nan"), "recall_at_k": float("nan"),
                "best_threshold": 0.5}

    out: Dict[str, float] = {}
    out["auroc"] = float(roc_auc_score(y_true, y_score))
    out["auprc"] = float(average_precision_score(y_true, y_score))

    thresholds = np.unique(y_score)
    if len(thresholds) > 200:
        thresholds = np.quantile(y_score, np.linspace(0.01, 0.99, 200))
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    out["f1"] = float(best_f1)
    out["best_threshold"] = best_t

    n_pos = max(1, int(y_true.sum()))
    k_eff = min(k, len(y_score))
    top_k = np.argsort(y_score)[-k_eff:]
    out["precision_at_k"] = float(y_true[top_k].mean())
    out["recall_at_k"] = float(y_true[top_k].sum() / n_pos)

    # Mordensky 2025 head-to-head metrics — capture rate at top X% of cells.
    # The 2025 paper reports 85% capture in the top 10% for the Great Basin.
    n = len(y_score)
    for pct in (0.01, 0.05, 0.10, 0.25):
        n_top = max(1, int(np.ceil(pct * n)))
        top_idx = np.argpartition(y_score, -n_top)[-n_top:]
        out[f"capture_top_{int(pct*100)}pct"] = float(y_true[top_idx].sum() / n_pos)
    return out


# ---------------------------------------------------------------------------
@torch.no_grad()
def build_z_lookup(model: torch.nn.Module, indices: np.ndarray,
                   processed: Path, use_thermal: bool, batch_size: int,
                   device: str, fused_dim: int) -> torch.Tensor:
    """Run the encoder + fusion path over `indices`, return [N_total, fused_dim]
    keyed by cell_id. Cells not in `indices` keep their zero row — they are
    never queried as neighbours during this fold.
    """
    n_total = int(np.load(processed / "labels.npy", mmap_mode="r").shape[0])
    lookup = torch.zeros(n_total, fused_dim, device=device)
    ds = GeoProspectDataset(processed, indices=indices, use_thermal=use_thermal)
    loader = make_loader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    for batch in loader:
        cell_ids = batch["cell_id"].to(device)
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        out = model(batch_dev)
        lookup[cell_ids] = out["z_fused"]
    return lookup


@torch.no_grad()
def evaluate_with_smoothing(model: torch.nn.Module, val_idx: np.ndarray,
                            z_lookup: torch.Tensor,
                            neighbor_indices: np.ndarray,
                            processed: Path, use_thermal: bool,
                            batch_size: int, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Re-score `val_idx` with spatial-smoothed embeddings."""
    ds = GeoProspectDataset(processed, indices=val_idx, use_thermal=use_thermal)
    loader = make_loader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    y_true, y_score = [], []
    for batch in loader:
        cell_ids = batch["cell_id"].numpy()
        neighbor_cells = neighbor_indices[cell_ids]  # [B, k]
        neighbor_z = z_lookup[torch.as_tensor(neighbor_cells, device=device,
                                              dtype=torch.long)]
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        out = model(batch_dev, neighbor_embeddings=neighbor_z)
        y_score.append(torch.sigmoid(out["logits"]).cpu().numpy())
        y_true.append(batch["label"].cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_score)


# ---------------------------------------------------------------------------
def train_one_split(cfg: Dict, train_idx: np.ndarray, val_idx: np.ndarray,
                    seed: int, device: str, log_prefix: str = "") -> Tuple[Dict, Path]:
    seed_everything(seed)
    processed = ROOT / cfg["paths"]["processed_dir"]
    ckpt_dir = ROOT / cfg["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_thermal = cfg["modalities"]["use_thermal"]
    bs = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    train_ds = GeoProspectDataset(processed, indices=train_idx, use_thermal=use_thermal)
    val_ds = GeoProspectDataset(processed, indices=val_idx, use_thermal=use_thermal)

    train_loader = make_loader(
        train_ds, batch_size=bs, shuffle=True, num_workers=num_workers,
        oversample_positive=cfg["training"]["positive_oversampling"],
    )
    val_loader = make_loader(
        val_ds, batch_size=bs, shuffle=False, num_workers=num_workers,
    )

    model = GeoProspectNet(cfg).to(device)
    # If oversampling is on, the focal alpha would double-weight positives;
    # neutralise alpha in that case so the two mechanisms don't compound.
    focal_alpha = cfg["training"]["focal_alpha"]
    if cfg["training"]["positive_oversampling"] > 1.0:
        focal_alpha = 0.5
    margin_weight = cfg["training"].get("margin_weight", 0.0)
    if margin_weight > 0:
        focal = CombinedLoss(
            focal_alpha=focal_alpha,
            focal_gamma=cfg["training"]["focal_gamma"],
            margin_pos=cfg["training"].get("margin_pos", 0.7),
            margin_neg=cfg["training"].get("margin_neg", 0.3),
            margin_weight=margin_weight,
        ).to(device)
    else:
        focal = FocalLoss(alpha=focal_alpha,
                          gamma=cfg["training"]["focal_gamma"]).to(device)

    contrastive_w = cfg["training"]["contrastive_weight"]
    opt = AdamW(model.parameters(),
                lr=cfg["training"]["learning_rate"],
                weight_decay=cfg["training"]["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=cfg["training"]["scheduler_T_max"],
                              eta_min=cfg["training"]["scheduler_eta_min"])

    # Pre-load neighbour table — used for smoothed validation each epoch.
    neighbor_indices = np.load(processed / "neighbor_indices.npy")
    labeled_for_lookup = np.unique(np.concatenate([train_idx, val_idx]))
    fused_dim = cfg["model"]["fused_dim"]

    best = {"epoch": -1, "auroc": -1.0}
    patience = cfg["training"]["early_stopping_patience"]
    bad_epochs = 0
    ckpt_path = ckpt_dir / f"{log_prefix}seed{seed}_best.pt"
    history: List[Dict] = []

    # Always materialise the first checkpoint so downstream loaders have
    # something to read, even if validation never produces a finite AUROC.
    save_checkpoint({"state_dict": model.state_dict(), "config": cfg,
                     "epoch": -1, "metrics": {"auroc": float("nan")}},
                    ckpt_path)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(batch)
            loss_focal = focal(out["logits"], batch["label"])
            loss = loss_focal + contrastive_w * out["l_contrastive"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg["training"]["grad_clip"])
            opt.step()
            running += float(loss.detach())
        sched.step()
        train_loss = running / max(1, len(train_loader))

        # ---- Smoothed validation -----------------------------------------
        if cfg["model"]["spatial_alpha"] > 0:
            z_lookup = build_z_lookup(model, labeled_for_lookup, processed,
                                      use_thermal, bs, device, fused_dim)
            y_true, y_score = evaluate_with_smoothing(
                model, val_idx, z_lookup, neighbor_indices,
                processed, use_thermal, bs, device,
            )
        else:
            model.eval()
            y_score, y_true = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    out = model(batch)
                    y_score.append(torch.sigmoid(out["logits"]).cpu().numpy())
                    y_true.append(batch["label"].cpu().numpy())
            y_true = np.concatenate(y_true); y_score = np.concatenate(y_score)

        metrics = metrics_at_threshold(y_true, y_score, k=50)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        # Use a finite floor so a NaN AUROC never wins the comparison.
        improved = (not np.isnan(metrics["auroc"])
                    and metrics["auroc"] > best["auroc"])
        if improved:
            best = {"epoch": epoch, **metrics}
            save_checkpoint({"state_dict": model.state_dict(),
                             "config": cfg, "epoch": epoch,
                             "metrics": metrics}, ckpt_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    return {"best": best, "history": history, "ckpt": str(ckpt_path)}, ckpt_path


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="Run first 5 LOF folds only")
    parser.add_argument("--all_folds", action="store_true",
                        help="Run all LOF-CV folds")
    parser.add_argument("--fold", type=int, default=None, help="Run one specific fold")
    parser.add_argument("--random", action="store_true",
                        help="Run the random 70/15/15 development split instead")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    print(f"device: {device}")

    splits_dir = ROOT / cfg["paths"]["splits_dir"]
    results_dir = ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.random:
        split = json.load(open(splits_dir / "train_val_test.json"))
        train_idx = np.array(split["train_idx"])
        val_idx = np.array(split["val_idx"])
        seeds = [args.seed] if args.seed is not None else list(range(42, 42 + cfg["training"]["n_seeds"]))
        all_results = []
        for s in seeds:
            r, _ = train_one_split(cfg, train_idx, val_idx, seed=s, device=device,
                                   log_prefix="random_")
            all_results.append(r)
        write_json(results_dir / "random_split_results.json", all_results)
        return

    folds = json.load(open(splits_dir / "lof_cv_folds.json"))
    if args.fold is not None:
        folds = [folds[args.fold]]
    elif args.quick:
        folds = folds[:5]
    elif not args.all_folds:
        folds = folds[:1]

    all_results: List[Dict] = []
    for i, fold in enumerate(folds):
        prefix = f"fold{fold['field_id']}_"
        print(f"\n=== Fold {i + 1}/{len(folds)}  (field {fold['field_id']}) ===")
        r, _ = train_one_split(
            cfg,
            np.array(fold["train_idx"]),
            np.array(fold["test_idx"]),
            seed=args.seed or 42,
            device=device,
            log_prefix=prefix,
        )
        r["fold_id"] = fold["field_id"]
        all_results.append(r)
        write_json(results_dir / "lof_cv_results.json", all_results)
        print(f"  fold {fold['field_id']} best AUROC={r['best']['auroc']:.4f}")

    aurocs = [r["best"]["auroc"] for r in all_results
              if not np.isnan(r["best"]["auroc"])]
    if aurocs:
        print(f"\nLOF-CV mean AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")


if __name__ == "__main__":
    main()
