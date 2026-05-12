"""Figure 4 — the headline continental prospectivity map.

Plots p(geothermal) across the western US with known fields and discovered
sites overlaid. Falls back to a plain matplotlib map if cartopy is unavailable
so the script works on minimal environments (Kaggle, CI).
"""
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


def _make_axes(extent):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig = plt.figure(figsize=(12, 9))
        ax = plt.axes(projection=ccrs.AlbersEqualArea(
            central_longitude=-114, central_latitude=40))
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.4)
        ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.5)
        return fig, ax, ccrs.PlateCarree()
    except ImportError:
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        return fig, ax, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    sr = cfg["study_region"]
    processed = ROOT / cfg["paths"]["processed_dir"]
    results = ROOT / cfg["paths"]["results_dir"]
    figs = ROOT / cfg["paths"]["figures_dir"]
    figs.mkdir(parents=True, exist_ok=True)

    grid_df = pd.read_csv(processed / "grid_coordinates.csv")
    mean_p = np.load(results / "prospectivity_mean.npy")

    n_rows = int(grid_df["row"].max()) + 1
    n_cols = int(grid_df["col"].max()) + 1
    img = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    img[grid_df["row"].values, grid_df["col"].values] = mean_p

    extent = [sr["lon_min"], sr["lon_max"], sr["lat_min"], sr["lat_max"]]
    fig, ax, transform = _make_axes(extent)
    cmap = plt.cm.get_cmap("RdYlBu_r")
    if transform is not None:
        im = ax.imshow(img[::-1, :], extent=extent, origin="upper",
                       cmap=cmap, vmin=0, vmax=1, transform=transform, alpha=0.85)
    else:
        im = ax.imshow(img[::-1, :], extent=extent, origin="upper",
                       cmap=cmap, vmin=0, vmax=1, alpha=0.85)

    fields = pd.read_csv(ROOT / cfg["paths"]["metadata_dir"] / "known_fields_details.csv")
    if transform is not None:
        ax.scatter(fields["lon"], fields["lat"], marker="^", c="black",
                   s=30, transform=transform, label="Known field", edgecolor="white",
                   linewidth=0.5)
    else:
        ax.scatter(fields["lon"], fields["lat"], marker="^", c="black",
                   s=30, label="Known field", edgecolor="white", linewidth=0.5)

    disc_path = results / "table3_discoveries.csv"
    if disc_path.exists():
        disc = pd.read_csv(disc_path)
        if len(disc):
            if transform is not None:
                ax.scatter(disc["lon"], disc["lat"], marker="*", c="yellow",
                           edgecolor="black", s=160, transform=transform,
                           label="Discovered", linewidth=0.7)
            else:
                ax.scatter(disc["lon"], disc["lat"], marker="*", c="yellow",
                           edgecolor="black", s=160, label="Discovered", linewidth=0.7)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("p(geothermal)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_title("GeoProspectNet — Continental Geothermal Prospectivity (Western US)")

    out = figs / "figure4_prospectivity_map.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
