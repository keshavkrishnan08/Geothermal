"""Audit whether LOFCV cluster boundaries (10 km) overlap with the
25 km discovery exclusion buffer in a way that could let co-genetic positives
leak into the same fold.

For each LOFCV fold, list the discovery centroids within 25 km of any
training cell in that fold. If a discovery centroid is within 25 km of
training cells from MULTIPLE folds, it suggests the 10 km field-clustering
is too tight and co-genetic positives have been split.

Outputs ``outputs/results/lofcv_buffer_audit.csv``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]


def main():
    folds_path = ROOT / "data/processed/splits/lof_cv_folds.json"
    if not folds_path.exists():
        print(f"[fatal] {folds_path} missing")
        return
    folds = json.load(open(folds_path))
    grid = pd.read_csv(ROOT / "data/processed/grid_coordinates.csv")
    discoveries = pd.read_csv(ROOT / "outputs/results/consensus_with_mwe_v2.csv")

    lat0 = float(grid.lat.mean())
    cos_lat0 = float(np.cos(np.radians(lat0)))
    gxy = np.column_stack([grid.lon * 111.32 * cos_lat0, grid.lat * 111.32])
    dxy = np.column_stack([discoveries.lon * 111.32 * cos_lat0,
                            discoveries.lat * 111.32])

    rows = []
    fold_membership = {int(d.discovery_id): set() for _, d in discoveries.iterrows()}
    for fi, fold in enumerate(folds):
        # 'train_idx' may be present or fold may be a dict with 'test_idx' meaning the fold-out cluster
        train_idx = np.asarray(fold.get("train_idx", []))
        if len(train_idx) == 0:
            continue
        train_xy = gxy[train_idx]
        tree = cKDTree(train_xy)
        d, _ = tree.query(dxy, k=1)
        for di, dist in enumerate(d):
            if dist <= 25.0:
                fold_membership[int(discoveries.iloc[di].discovery_id)].add(fi)

    rows = []
    for d_id, folds_touched in fold_membership.items():
        rows.append({
            "discovery_id": d_id,
            "n_folds_within_25km": len(folds_touched),
            "fold_indices": sorted(list(folds_touched))[:10],
        })
    df = pd.DataFrame(rows).sort_values("n_folds_within_25km", ascending=False)
    out = ROOT / "outputs/results/lofcv_buffer_audit.csv"
    df.to_csv(out, index=False)

    print("=== LOFCV ↔ 25 km discovery-buffer audit ===")
    print(f"folds inspected: {len(folds)}")
    n_with_overlap = int((df.n_folds_within_25km > 0).sum())
    n_single_fold = int((df.n_folds_within_25km == 1).sum())
    n_multi_fold = int((df.n_folds_within_25km > 1).sum())
    print(f"discoveries with ≥1 fold within 25 km: {n_with_overlap}/{len(df)}")
    print(f"  exactly 1 fold (clean):             {n_single_fold}")
    print(f"  multiple folds (co-genetic risk):   {n_multi_fold}")
    print()
    print("Interpretation: a discovery within 25 km of *multiple* LOFCV folds")
    print("suggests the 10 km field-clustering may be too tight and would have")
    print("split co-genetic positives across folds, allowing leakage.")
    print()
    print(df.head(10).to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
