"""Compare GeoProspectNet's continental scores against Mordensky et al. (2023)'s
seven published favorability surfaces (LR, ensemble-LR, XGB, ensemble-XGB,
SVM, ensemble-SVM, ANN).

For each method we report:
  - Post-2008 hold-out mean percentile + top-10% capture
  - NC3 random-deep mean percentile (should be low)
  - Spatial agreement with GeoProspectNet's top 1% (IoU)

Mordensky covers only the Great Basin; we restrict the comparison to cells
inside their domain. Outputs ``outputs/results/mordensky_comparison.csv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]

MORDENSKY_METHODS = ["LR", "enLR", "XGB", "enXGB", "SVM", "enSVM", "ANN"]


def main():
    mord = pd.read_csv(ROOT / "data/raw/labels/mordensky2023_full.csv")
    mord = mord.dropna(subset=["lat_83", "lon_83"]).reset_index(drop=True)
    print(f"Mordensky 2023 cells: {len(mord):,}")

    grid = pd.read_csv(ROOT / "data/processed/grid_coordinates.csv")
    scores_gpn = np.load(ROOT / "outputs/results/scores_cpu_max.npy")

    # Project to km
    lat0 = float(grid.lat.mean())
    cos_lat0 = float(np.cos(np.radians(lat0)))
    grid_xy = np.column_stack([grid.lon * 111.32 * cos_lat0, grid.lat * 111.32])
    mord_xy = np.column_stack([mord.lon_83 * 111.32 * cos_lat0, mord.lat_83 * 111.32])

    # Each grid cell → nearest Mordensky cell. Drop matches > 8 km (outside domain).
    tree = cKDTree(mord_xy)
    d, idx = tree.query(grid_xy, k=1)
    in_domain = d < 8.0
    print(f"Grid cells inside Mordensky domain: {in_domain.sum():,} of {len(grid):,}")

    # Restrict everything to in-domain cells
    grid_d = grid[in_domain].reset_index(drop=True)
    scores_gpn_d = scores_gpn[in_domain]
    mord_at_grid = mord.iloc[idx[in_domain]].reset_index(drop=True)

    # Hold-out
    ho = pd.read_csv(ROOT / "data/raw/labels/nv_permits_holdout.csv")
    novel = ho[ho.is_novel].reset_index(drop=True) if "is_novel" in ho.columns else ho
    domain_xy = np.column_stack([grid_d.lon * 111.32 * cos_lat0,
                                  grid_d.lat * 111.32])
    tree_d = cKDTree(domain_xy)
    ho_idx = []
    for _, r in novel.iterrows():
        d, i = tree_d.query([r.lon * 111.32 * cos_lat0, r.lat * 111.32], k=1)
        if d < 8.0:  # only count hold-outs inside Mordensky domain
            ho_idx.append(i)
    ho_idx = np.array(ho_idx)
    print(f"Hold-out sites inside Mordensky domain: {len(ho_idx)} of {len(novel)}")

    # NC3 random deep within domain
    fields = pd.read_csv(ROOT / "data/metadata/known_fields_details.csv")
    fxy = np.column_stack([fields.lon * 111.32 * cos_lat0, fields.lat * 111.32])
    d_to_field, _ = cKDTree(fxy).query(domain_xy, k=1)
    masks = np.load(ROOT / "data/processed/modality_masks.npy")[in_domain]
    nc3 = (d_to_field > 100) & (~masks[:, 1])
    rng = np.random.default_rng(42)
    nc3_idx = rng.choice(np.flatnonzero(nc3), size=min(100, int(nc3.sum())), replace=False)

    methods = {"GeoProspectNet": scores_gpn_d}
    for m in MORDENSKY_METHODS:
        methods[m] = mord_at_grid[m].values

    rows = []
    for name, s in methods.items():
        # rank scores ascending → percentile 0–100 within Mordensky domain
        sort = np.argsort(s)
        rank = np.empty_like(sort, dtype=np.int64)
        rank[sort] = np.arange(len(s))
        pct = 100.0 * rank / (len(s) - 1)
        if len(ho_idx) > 0:
            ho_pcts = pct[ho_idx]
            ho_mean = float(ho_pcts.mean())
            ho_top10 = float((ho_pcts >= 90).mean())
        else:
            ho_mean = float("nan"); ho_top10 = float("nan")
        nc3_pcts = pct[nc3_idx]
        nc3_mean = float(nc3_pcts.mean())
        # IoU of top 1% with GeoProspectNet
        top1_gpn = set(np.argsort(-scores_gpn_d)[:max(1, len(s) // 100)])
        top1_m = set(np.argsort(-s)[:max(1, len(s) // 100)])
        iou = len(top1_gpn & top1_m) / max(1, len(top1_gpn | top1_m))
        rows.append(dict(method=name,
                          holdout_mean_pct=ho_mean,
                          holdout_top10_capture=ho_top10,
                          nc3_mean_pct=nc3_mean,
                          iou_top1_vs_GPN=iou))

    df = pd.DataFrame(rows)
    out = ROOT / "outputs/results/mordensky_comparison.csv"
    df.to_csv(out, index=False)
    print("\n=== GeoProspectNet vs Mordensky 2023 (Great Basin sub-domain) ===")
    print(df.to_string(index=False, float_format="%.3f"))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
