"""Train baselines B1..B7 on the same LOF-CV folds as GeoProspectNet.

For B1-B4 we use the flat-feature vector. For B5 we train a 4-layer MLP on
the same flat features. For B6 we set ``contrastive_weight=0`` and re-train
the full deep model. For B7 we train one model per modality and report the best.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import GeoProspectDataset, make_loader
from src.models.baselines import (
    FlatMLP, build_flat_features, make_late_fusion_config, make_lgbm,
    make_logreg, make_rf, make_single_modality_config, make_xgb,
)
from src.training.losses import FocalLoss
from src.training.train import train_one_split, metrics_at_threshold
from src.training.utils import device_auto, load_config, seed_everything, write_json

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
def fit_flat(model_factory: Callable, X: np.ndarray, y: np.ndarray,
             train_idx: np.ndarray, test_idx: np.ndarray) -> Dict:
    """Fit a sklearn / xgboost / lgbm model and return fold metrics."""
    model = model_factory()
    model.fit(X[train_idx], y[train_idx])
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X[test_idx])[:, 1]
    else:  # SVMs etc.
        scores = model.decision_function(X[test_idx])
    return metrics_at_threshold(y[test_idx], scores, k=50)


def fit_mlp(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
            epochs: int = 80, batch_size: int = 256, lr: float = 1e-3,
            device: str = "cpu", seed: int = 42) -> Dict:
    seed_everything(seed)
    in_f = X.shape[1]
    model = FlatMLP(in_features=in_f).to(device)
    focal = FocalLoss().to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    Xt = torch.from_numpy(X[train_idx]).float().to(device)
    yt = torch.from_numpy(y[train_idx]).float().to(device)
    Xv = torch.from_numpy(X[test_idx]).float().to(device)
    yv = y[test_idx]

    # Mini-batch loop
    perm = torch.randperm(len(Xt))
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(Xt[idx])
            loss = focal(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(Xv)).cpu().numpy()
    return metrics_at_threshold(yv, scores, k=50)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument("--baseline", type=str, default="all",
                        choices=["all", "logreg", "rf", "xgb", "lgbm", "mlp", "late_fusion", "single"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    processed = ROOT / cfg["paths"]["processed_dir"]
    splits_dir = ROOT / cfg["paths"]["splits_dir"]
    results_dir = ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    folds = json.load(open(splits_dir / "lof_cv_folds.json"))
    if args.quick:
        folds = folds[:5]
    elif not args.all_folds:
        folds = folds[:1]

    print("Building flat features ...")
    X = build_flat_features(processed, use_thermal=cfg["modalities"]["use_thermal"])
    y = np.load(processed / "labels.npy").astype(np.float32)
    train_mask = np.load(processed / "train_mask.npy")
    print(f"  X shape: {X.shape}, positives: {int(y[train_mask].sum())}")

    factories = {
        "logreg": ("Logistic Regression", make_logreg),
        "rf": ("Random Forest", make_rf),
        "xgb": ("XGBoost", make_xgb),
        "lgbm": ("LightGBM", make_lgbm),
    }
    selected = list(factories.keys()) if args.baseline == "all" else [args.baseline]

    results: Dict[str, List[Dict]] = {}

    # ---- flat-feature baselines ------------------------------------------
    for key in selected:
        if key not in factories:
            continue
        name, factory = factories[key]
        per_fold = []
        for fold in folds:
            train_idx = np.array(fold["train_idx"])
            test_idx = np.array(fold["test_idx"])
            try:
                m = fit_flat(factory, X, y, train_idx, test_idx)
            except Exception as e:
                print(f"[warn] {name} fold {fold['field_id']}: {e}")
                m = {"auroc": float("nan")}
            m["fold_id"] = fold["field_id"]
            per_fold.append(m)
            print(f"  {name:20s}  fold {fold['field_id']:4d}  AUROC={m.get('auroc', float('nan')):.4f}")
        results[name] = per_fold

    # ---- B5 MLP ----------------------------------------------------------
    if args.baseline in ("all", "mlp"):
        per_fold = []
        for fold in folds:
            m = fit_mlp(X, y, np.array(fold["train_idx"]),
                        np.array(fold["test_idx"]), device=device)
            m["fold_id"] = fold["field_id"]
            per_fold.append(m)
            print(f"  Flat MLP             fold {fold['field_id']:4d}  AUROC={m['auroc']:.4f}")
        results["Flat MLP"] = per_fold

    # ---- B6 Late Fusion (no contrastive) --------------------------------
    if args.baseline in ("all", "late_fusion"):
        lf_cfg = make_late_fusion_config(cfg)
        per_fold = []
        for fold in folds:
            r, _ = train_one_split(
                lf_cfg, np.array(fold["train_idx"]),
                np.array(fold["test_idx"]), seed=42, device=device,
                log_prefix=f"latefusion_fold{fold['field_id']}_",
            )
            m = r["best"]
            m["fold_id"] = fold["field_id"]
            per_fold.append(m)
            print(f"  Late Fusion MLP      fold {fold['field_id']:4d}  AUROC={m['auroc']:.4f}")
        results["Late Fusion MLP"] = per_fold

    # ---- B7 Single-modality (best of 4) ---------------------------------
    if args.baseline in ("all", "single"):
        modalities = ["geophysics", "geochemistry", "geology"]
        if cfg["modalities"]["use_thermal"]:
            modalities.append("thermal")
        per_fold_best, per_keep_summary = [], {m: [] for m in modalities}
        for fold in folds:
            best_auc, best_mod = float("-inf"), None
            best_metrics: Dict = {}
            for keep in modalities:
                # GeoProspectNet supports a LIST of dropped modalities; we
                # drop everything except `keep`, giving a faithful
                # single-modality baseline that reuses the same architecture
                # and trainer (with contrastive disabled — no pairs).
                sm_cfg = copy.deepcopy(cfg)
                sm_cfg["modalities"]["drop"] = [m for m in modalities if m != keep]
                sm_cfg["training"]["contrastive_weight"] = 0.0
                r, _ = train_one_split(
                    sm_cfg, np.array(fold["train_idx"]),
                    np.array(fold["test_idx"]), seed=42, device=device,
                    log_prefix=f"single_{keep}_fold{fold['field_id']}_",
                )
                auc = r["best"]["auroc"]
                per_keep_summary[keep].append(auc)
                if not np.isnan(auc) and auc > best_auc:
                    best_auc, best_mod = auc, keep
                    best_metrics = {**r["best"], "modality": keep}
            best_metrics["fold_id"] = fold["field_id"]
            per_fold_best.append(best_metrics)
            print(f"  Best Single-Modality fold {fold['field_id']:4d} "
                  f"({best_mod})  AUROC={best_auc:.4f}")
        results["Best Single-Modality"] = per_fold_best
        write_json(results_dir / "single_modality_per_keep.json", per_keep_summary)

    write_json(results_dir / "baseline_results.json", results)
    print(f"\nWrote {results_dir / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
