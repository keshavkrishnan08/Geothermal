"""All seven baseline models (B1..B7) plus a flat-feature builder.

Flat features per cell (Section 3 of CLAUDE.md):
    geophysics  30  = per-channel [mean, std, min, max, median] over 32x32
    geochemistry 9  = raw 9-d vector
    thermal      7  = [mean, std, min, max, median, p90, p10] over 64x64
    geology     11  = raw 11-d vector
    total       57
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# B1-B4 use scikit-learn / xgboost / lightgbm — imported lazily so missing
# packages don't crash the unit tests for the deep model.


def patch_stats(patches: np.ndarray, channel_axis: int = 1,
                quantiles: bool = False) -> np.ndarray:
    """Compute per-channel summary statistics over the spatial dimensions."""
    flat = patches.reshape(patches.shape[0], patches.shape[channel_axis], -1)
    stats = [flat.mean(-1), flat.std(-1), flat.min(-1), flat.max(-1),
             np.median(flat, axis=-1)]
    if quantiles:
        stats += [np.percentile(flat, 90, axis=-1),
                  np.percentile(flat, 10, axis=-1)]
    return np.concatenate(stats, axis=-1)


def build_flat_features(processed: Path, use_thermal: bool = True) -> np.ndarray:
    """Return a [N, 57] (or [N, 50] without thermal) feature matrix."""
    geo = np.load(processed / "geophysics_patches.npy", mmap_mode="r")
    chem = np.load(processed / "geochemistry_features.npy", mmap_mode="r")
    struct = np.load(processed / "geology_features.npy", mmap_mode="r")

    geo_stats = patch_stats(geo, channel_axis=1, quantiles=False)
    chem_arr = np.asarray(chem)
    struct_arr = np.asarray(struct)

    if use_thermal:
        therm_path = processed / "thermal_patches.npy"
        therm = np.load(therm_path, mmap_mode="r") if therm_path.exists() else None
        if therm is not None:
            therm_stats = patch_stats(therm, channel_axis=1, quantiles=True)
        else:
            therm_stats = np.zeros((len(geo_stats), 7), dtype=np.float32)
        return np.concatenate([geo_stats, chem_arr, therm_stats, struct_arr], axis=1).astype(np.float32)
    return np.concatenate([geo_stats, chem_arr, struct_arr], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# B1 Logistic Regression
def make_logreg():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, penalty="l2", class_weight="balanced",
                              max_iter=1000, n_jobs=-1)


# B2 Random Forest
def make_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=500, max_depth=20, class_weight="balanced",
        min_samples_leaf=5, n_jobs=-1, random_state=42,
    )


# B3 XGBoost
def make_xgb():
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.1,
        scale_pos_weight=10, subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", n_jobs=-1, random_state=42,
        tree_method="hist",
    )


# B4 LightGBM
def make_lgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.1,
        scale_pos_weight=10, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, verbose=-1,
    )


# B5 MLP (flat features)
class FlatMLP(nn.Module):
    """Plain MLP over the 57-d flat-feature vector — B5 baseline."""

    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# B6 Late Fusion MLP (full deep encoders, NO contrastive loss).
# Implemented by setting `training.contrastive_weight: 0` in config and the
# attention layer still active — that's effectively "deep late-fusion MLP".
# We expose a small helper just to make the intent explicit.
def make_late_fusion_config(base: dict) -> dict:
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    cfg["training"]["contrastive_weight"] = 0.0
    return cfg


# B7 Single-modality models. Each is a copy of GeoProspectNet with all but
# one modality dropped (and contrastive disabled — no pairs to contrast).
def make_single_modality_config(base: dict, keep: str) -> dict:
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    cfg["training"]["contrastive_weight"] = 0.0
    drop_options = [m for m in ("geophysics", "geochemistry", "thermal", "geology") if m != keep]
    # We can only "drop" one modality at a time via the GPN config knob, so we
    # train a separate model per modality. This helper is a thin façade.
    cfg["modalities"]["_keep_only"] = keep
    return cfg
