"""Classical ML baselines on the same 70/15/15 split used by GeoProspectNet.

Compared methods:
  - Heat-flow threshold (>80 mW/m^2 z-scored → label=1)
  - Logistic regression
  - Random Forest (200 trees)
  - XGBoost (200 trees, depth 6)

Features: mean-pooled geophysics patches (6) + geochem (9) + geology (11) = 26.

Reports AUROC, AUPRC, capture@5%, capture@10%, plus the post-2008 hold-out
mean percentile. Writes outputs/results/baselines.csv and prints the table.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]


def _build_features(processed: Path) -> np.ndarray:
    geophys = np.load(processed / "geophysics_patches.npy", mmap_mode="r")  # [N,6,32,32]
    geochem = np.load(processed / "geochemistry_features.npy", mmap_mode="r")  # [N,9]
    geology = np.load(processed / "geology_features.npy", mmap_mode="r")      # [N,11]
    # Mean-pool patches for tabular ML
    gp = geophys.mean(axis=(2, 3)).astype(np.float32)   # [N,6]
    return np.concatenate([gp, np.asarray(geochem, dtype=np.float32),
                           np.asarray(geology, dtype=np.float32)], axis=1)


def _percentile(scores: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.array([100.0 * (scores < scores[i]).mean() for i in idx])


def _capture_at_k(y_true: np.ndarray, scores: np.ndarray, k_frac: float) -> float:
    n_top = max(1, int(len(scores) * k_frac))
    top = np.argsort(-scores)[:n_top]
    return float(y_true[top].sum() / max(1, y_true.sum()))


def _holdout_pct(scores_full: np.ndarray, grid_df: pd.DataFrame,
                 holdout_path: Path) -> tuple[float, float]:
    hp = pd.read_csv(holdout_path)
    novel = hp[hp.is_novel].reset_index(drop=True) if "is_novel" in hp.columns else hp
    lat0 = float(grid_df.lat.mean()); cos_lat0 = float(np.cos(np.radians(lat0)))
    gxy = np.column_stack([grid_df.lon * 111.32 * cos_lat0, grid_df.lat * 111.32])
    tree = cKDTree(gxy)
    pcts = []
    for _, r in novel.iterrows():
        _, i = tree.query([r.lon * 111.32 * cos_lat0, r.lat * 111.32], k=1)
        pcts.append(100.0 * (scores_full < scores_full[i]).mean())
    pcts = np.array(pcts)
    return float(pcts.mean()), float((pcts >= 90).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_max.yaml")
    parser.add_argument("--holdout", default="data/raw/labels/nv_permits_holdout.csv")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))

    processed = ROOT / cfg["paths"]["processed_dir"]
    print("Loading features ...", flush=True)
    X = _build_features(processed)
    labels = np.load(processed / "labels.npy")
    train_mask = np.load(processed / "train_mask.npy")
    grid_df = pd.read_csv(processed / "grid_coordinates.csv")
    n_total = len(X)

    # 70/15/15 split among labeled cells
    labeled = np.flatnonzero(train_mask)
    y = labels[labeled]
    Xl = X[labeled]
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        Xl, y, labeled, test_size=0.30, stratify=y, random_state=42)
    X_va, X_te, y_va, y_te, idx_va, idx_te = train_test_split(
        X_te, y_te, idx_te, test_size=0.50, stratify=y_te, random_state=42)
    print(f"train={len(X_tr)} val={len(X_va)} test={len(X_te)}")

    # Standardize for LR
    mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-6

    methods = {}
    try:
        from xgboost import XGBClassifier
        methods["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1, verbosity=0,
            scale_pos_weight=(y_tr == 0).sum() / max(1, (y_tr == 1).sum()),
            eval_metric="logloss",
        )
    except ImportError:
        print("[warn] xgboost not installed; skipping")

    methods["RandomForest"] = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, class_weight="balanced",
        random_state=42,
    )
    methods["LogisticRegression"] = LogisticRegression(
        max_iter=500, class_weight="balanced", n_jobs=-1, random_state=42,
    )

    rows = []
    full_scores = {}
    for name, clf in methods.items():
        print(f"-> training {name} ...", flush=True)
        if name == "LogisticRegression":
            clf.fit((X_tr - mu) / sd, y_tr)
            s_te = clf.predict_proba((X_te - mu) / sd)[:, 1]
            s_full = clf.predict_proba((X - mu) / sd)[:, 1]
        else:
            clf.fit(X_tr, y_tr)
            s_te = clf.predict_proba(X_te)[:, 1]
            s_full = clf.predict_proba(X)[:, 1]
        auroc = roc_auc_score(y_te, s_te)
        auprc = average_precision_score(y_te, s_te)
        cap5 = _capture_at_k(y_te, s_te, 0.05)
        cap10 = _capture_at_k(y_te, s_te, 0.10)
        h_mean, h_top10 = _holdout_pct(s_full, grid_df, ROOT / args.holdout)
        rows.append(dict(method=name, auroc=auroc, auprc=auprc,
                         capture_top5pct=cap5, capture_top10pct=cap10,
                         holdout_mean_pct=h_mean, holdout_top10_capture=h_top10))
        full_scores[name] = s_full
        print(f"   AUROC={auroc:.3f} AUPRC={auprc:.3f} cap@5={cap5:.2%} hold-out mean pct={h_mean:.1f}")

    # Heat-flow rule: take mean heat-flow channel value as score
    print("-> heat-flow threshold rule ...", flush=True)
    hf_score = X[:, 3]  # geophys channel 3 (heat flow) after mean-pool
    s_te_hf = hf_score[idx_te]
    rows.append(dict(
        method="HeatFlowRule",
        auroc=roc_auc_score(y_te, s_te_hf),
        auprc=average_precision_score(y_te, s_te_hf),
        capture_top5pct=_capture_at_k(y_te, s_te_hf, 0.05),
        capture_top10pct=_capture_at_k(y_te, s_te_hf, 0.10),
        holdout_mean_pct=_holdout_pct(hf_score, grid_df, ROOT / args.holdout)[0],
        holdout_top10_capture=_holdout_pct(hf_score, grid_df, ROOT / args.holdout)[1],
    ))

    # GeoProspectNet: load the cpu_max continental scores if cached
    s_cpu_max = ROOT / "outputs/results/scores_cpu_max.npy"
    if s_cpu_max.exists():
        s_full = np.load(s_cpu_max)
        s_te_dl = s_full[idx_te]
        rows.append(dict(
            method="GeoProspectNet",
            auroc=roc_auc_score(y_te, s_te_dl),
            auprc=average_precision_score(y_te, s_te_dl),
            capture_top5pct=_capture_at_k(y_te, s_te_dl, 0.05),
            capture_top10pct=_capture_at_k(y_te, s_te_dl, 0.10),
            holdout_mean_pct=_holdout_pct(s_full, grid_df, ROOT / args.holdout)[0],
            holdout_top10_capture=_holdout_pct(s_full, grid_df, ROOT / args.holdout)[1],
        ))

    df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    out = ROOT / "outputs/results/baselines.csv"
    df.to_csv(out, index=False)
    print()
    print("=== BASELINES vs GeoProspectNet ===")
    print(df.to_string(index=False, float_format="%.3f"))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
