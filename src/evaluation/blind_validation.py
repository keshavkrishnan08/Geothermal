"""Blind validation against contemporaneous commercial discoveries.

Reads ``data/metadata/zanskar_validation.csv`` (or any other CSV with name/lat/lon
columns) and reports:
    - the predicted favourability percentile of each named site
    - whether the site falls in the top {1, 5, 10}% of predicted favourability
    - distance to the nearest known field (sanity check that we didn't include it)

This is the headline ``did the model independently rank a real industry
discovery in its top tier?'' check.
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
    parser.add_argument("--validation_csv", default="data/metadata/zanskar_validation.csv")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))

    processed = ROOT / cfg["paths"]["processed_dir"]
    grid_df = pd.read_csv(processed / "grid_coordinates.csv")
    mean_p = np.load(ROOT / "outputs/results/prospectivity_mean.npy")
    std_p = np.load(ROOT / "outputs/results/prospectivity_std.npy")

    csv_path = ROOT / args.validation_csv
    if not csv_path.exists():
        print(f"[warn] {csv_path} missing — skipping blind validation")
        return
    sites = pd.read_csv(csv_path)

    # Match each validation site to its nearest grid cell
    lat0 = float(grid_df["lat"].mean())
    cos_lat0 = float(np.cos(np.radians(lat0)))
    grid_xy = np.column_stack([
        grid_df["lon"].values * 111.32 * cos_lat0,
        grid_df["lat"].values * 111.32,
    ])
    tree = cKDTree(grid_xy)

    rows = []
    for _, site in sites.iterrows():
        sx = float(site["lon"]) * 111.32 * cos_lat0
        sy = float(site["lat"]) * 111.32
        dist, idx = tree.query([sx, sy], k=1)
        cell_p = float(mean_p[idx])
        cell_std = float(std_p[idx])
        # Percentile of cell_p across the entire grid
        pct = float(100.0 * (mean_p < cell_p).mean())

        # Distance to nearest known field (was this site already in training?)
        meta = ROOT / cfg["paths"]["metadata_dir"]
        fields_path = meta / "known_fields_details.csv"
        nearest_field_km = float("nan")
        if fields_path.exists():
            fields = pd.read_csv(fields_path)
            if len(fields):
                f_xy = np.column_stack([
                    fields["lon"].values * 111.32 * cos_lat0,
                    fields["lat"].values * 111.32,
                ])
                ftree = cKDTree(f_xy)
                d, _ = ftree.query([sx, sy], k=1)
                nearest_field_km = float(d[0])

        rows.append({
            "name": site["name"],
            "lat": float(site["lat"]),
            "lon": float(site["lon"]),
            "predicted_p": cell_p,
            "predicted_std": cell_std,
            "percentile": pct,
            "in_top_1pct": pct >= 99.0,
            "in_top_5pct": pct >= 95.0,
            "in_top_10pct": pct >= 90.0,
            "nearest_known_field_km": nearest_field_km,
        })

    df = pd.DataFrame(rows)
    out = ROOT / "outputs/results/blind_validation.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
