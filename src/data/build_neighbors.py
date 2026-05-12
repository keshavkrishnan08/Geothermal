"""Pre-compute k-nearest-neighbor indices for spatial smoothing.

Output: data/processed/neighbor_indices.npy with shape [N, k] (int64).
Distances are computed in a metric (equirectangular km) projection so
neighbor selection is geographically meaningful.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    processed = ROOT / cfg["paths"]["processed_dir"]

    grid_df = pd.read_csv(processed / "grid_coordinates.csv")
    lats = grid_df["lat"].values
    lons = grid_df["lon"].values
    lat0 = float(lats.mean())
    cos_lat0 = float(np.cos(np.radians(lat0)))
    xy = np.column_stack([lons * 111.32 * cos_lat0, lats * 111.32])

    k = int(cfg["model"]["spatial_k_neighbors"]) + 1  # +1 because self is first
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k)
    idx = idx[:, 1:]  # drop self
    np.save(processed / "neighbor_indices.npy", idx.astype(np.int64))
    print(f"Wrote neighbor_indices.npy  shape={idx.shape}")


if __name__ == "__main__":
    main()
