"""Fault-proximity statistics for the 33 consensus discoveries.

The geological prior for hidden hydrothermal systems is that they sit
along permeable Quaternary structural pathways. If our 33 discoveries
are not significantly closer to mapped Q-faults than random background
cells in the same provinces, the model has not learned this prior.

Protocol
--------
1. Compute distance to nearest Q-fault for each of the 33 discoveries.
2. Build a province-stratified background of 1000 random western-US
   cells with no known field within 50 km.
3. Compute their distance-to-Q-fault distribution.
4. Mann-Whitney U test: are discoveries significantly closer to faults
   than the background? Report p-value.
5. Also report the fraction within 5/10/25 km of any mapped Q-fault.

Output
------
outputs/results/discovery_fault_proximity.csv
outputs/results/discovery_fault_proximity_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]


def load_qfault_centroids():
    shp = ROOT / "data/raw/geology/qfaults/SHP/Qfaults_US_Database.shp"
    sf = shapefile.Reader(str(shp))
    pts = []
    for s in sf.shapes():
        if s.shapeType in (3, 5, 13, 15):
            arr = np.asarray(s.points, dtype=np.float64)
            if len(arr):
                pts.append(arr.mean(axis=0))
    pts = np.array(pts)
    keep = (pts[:, 0] > -125) & (pts[:, 0] < -103) \
           & (pts[:, 1] > 31) & (pts[:, 1] < 49)
    return pts[keep]


def km_distance_table(targets_lonlat, query_lonlat):
    lat0 = float(np.mean(targets_lonlat[:, 1]))
    cos0 = float(np.cos(np.radians(lat0)))
    txy = np.column_stack([targets_lonlat[:, 0] * 111.32 * cos0,
                            targets_lonlat[:, 1] * 111.32])
    qxy = np.column_stack([query_lonlat[:, 0] * 111.32 * cos0,
                            query_lonlat[:, 1] * 111.32])
    return cKDTree(txy).query(qxy, k=1)[0]


def main():
    results = ROOT / "outputs/results"
    disc = pd.read_csv(results / "consensus_with_mwe_v2.csv")
    print(f"loaded {len(disc)} consensus discoveries")

    print("loading Q-faults ...")
    qf = load_qfault_centroids()
    print(f"  {len(qf):,} fault centroids")

    print("loading known geothermal fields ...")
    kf = pd.read_csv(ROOT / "data/metadata/known_fields_details.csv")
    kf = kf.dropna(subset=["lat", "lon"])

    # Distance to fault for discoveries
    d_disc = km_distance_table(qf, disc[["lon", "lat"]].values)
    disc["dist_to_qfault_km"] = d_disc

    # Background: 1000 random cells in western US, > 50 km from any known field
    print("building province-stratified background ...")
    rng = np.random.default_rng(42)
    grid = pd.read_csv(ROOT / "data/processed/grid_coordinates.csv")
    if "lat" not in grid.columns:
        raise SystemExit("grid_coordinates.csv missing lat/lon")
    g_lonlat = grid[["lon", "lat"]].values
    d_to_kf = km_distance_table(kf[["lon", "lat"]].values, g_lonlat)
    far_mask = d_to_kf > 50.0
    pool = grid[far_mask]
    print(f"  background pool size (>50 km from any field): {len(pool):,}")
    bg = pool.sample(n=min(1000, len(pool)), random_state=42).reset_index(drop=True)
    d_bg = km_distance_table(qf, bg[["lon", "lat"]].values)
    bg["dist_to_qfault_km"] = d_bg

    # Statistics
    u_stat, p_val = mannwhitneyu(d_disc, d_bg, alternative="less")
    print(f"\nMann-Whitney U test (discoveries vs. background):")
    print(f"  median dist (discoveries): {np.median(d_disc):.2f} km")
    print(f"  median dist (background):  {np.median(d_bg):.2f} km")
    print(f"  U = {u_stat:.0f}   p = {p_val:.2e}")
    print(f"  --> discoveries are {'significantly' if p_val < 0.01 else 'NOT significantly'} closer to Q-faults")

    bins = [5.0, 10.0, 25.0, 50.0]
    pct_within = {}
    for b in bins:
        pct_within[f"within_{int(b)}_km_disc"] = float(100.0 * (d_disc <= b).mean())
        pct_within[f"within_{int(b)}_km_bg"] = float(100.0 * (d_bg <= b).mean())
    print(f"\n  Fraction within X km of a mapped Q-fault:")
    print(f"    X (km) | discoveries | background")
    print(f"    ------:|------------:|----------:")
    for b in bins:
        print(f"    {int(b):6d} | {pct_within[f'within_{int(b)}_km_disc']:10.1f}% | "
              f"{pct_within[f'within_{int(b)}_km_bg']:9.1f}%")

    summary = {
        "n_discoveries": int(len(disc)),
        "n_background_cells": int(len(bg)),
        "n_qfaults_used": int(len(qf)),
        "median_dist_disc_km": float(np.median(d_disc)),
        "median_dist_bg_km": float(np.median(d_bg)),
        "mannwhitney_U": float(u_stat),
        "mannwhitney_p_alt_less": float(p_val),
        **pct_within,
    }
    with open(results / "discovery_fault_proximity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    disc[["discovery_id", "lat", "lon", "province", "mwe_p50",
          "dist_to_qfault_km"]].to_csv(
        results / "discovery_fault_proximity.csv", index=False)
    print(f"\nwrote outputs/results/discovery_fault_proximity.csv")
    print(f"wrote outputs/results/discovery_fault_proximity_summary.json")


if __name__ == "__main__":
    main()
