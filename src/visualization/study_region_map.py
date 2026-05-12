"""Figure 2 — study region with known fields and data coverage."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    sr = cfg["study_region"]
    processed = ROOT / cfg["paths"]["processed_dir"]
    meta = ROOT / cfg["paths"]["metadata_dir"]
    figs = ROOT / cfg["paths"]["figures_dir"]
    figs.mkdir(parents=True, exist_ok=True)

    grid_df = pd.read_csv(processed / "grid_coordinates.csv")
    masks = np.load(processed / "modality_masks.npy")
    coverage_count = masks.sum(axis=1)

    n_rows = int(grid_df["row"].max()) + 1
    n_cols = int(grid_df["col"].max()) + 1
    img = np.zeros((n_rows, n_cols))
    img[grid_df["row"].values, grid_df["col"].values] = coverage_count

    fig, ax = plt.subplots(figsize=(11, 8))
    extent = [sr["lon_min"], sr["lon_max"], sr["lat_min"], sr["lat_max"]]
    im = ax.imshow(img[::-1, :], extent=extent, origin="upper",
                   cmap="viridis", vmin=0, vmax=4, alpha=0.85)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Number of modalities available")

    fields_path = meta / "known_fields_details.csv"
    if fields_path.exists():
        fields = pd.read_csv(fields_path)
        ax.scatter(fields["lon"], fields["lat"], marker="^",
                   c="red", s=20, edgecolor="white", linewidth=0.4,
                   label="Known geothermal field")
    ax.set_title("Western US study region — data coverage and known fields")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(loc="lower left")

    out = figs / "figure2_study_region.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
