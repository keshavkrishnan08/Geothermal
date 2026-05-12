"""Sample 64x64 thermal anomaly patches from a GEE-exported GeoTIFF.

If the file is missing, the modality is gracefully skipped: thermal_patches.npy
is written as an all-zero placeholder and modality_masks.npy[:, 2] = False.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.data.grid import GridSpec, load_or_build_grid

ROOT = Path(__file__).resolve().parents[2]
PATCH = 64
THERMAL_RES_M = 100.0  # native pixel size of GEE export


def _read_window(ds, lon: float, lat: float, patch_px: int) -> np.ndarray:
    """Read a patch of `patch_px` pixels centered on (lon, lat)."""
    from rasterio.windows import Window
    try:
        row, col = ds.index(lon, lat)
    except Exception:
        return np.full((patch_px, patch_px), np.nan, dtype=np.float32)
    half = patch_px // 2
    win = Window(col - half, row - half, patch_px, patch_px)
    data = ds.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
    if data.shape != (patch_px, patch_px):
        out = np.full((patch_px, patch_px), np.nan, dtype=np.float32)
        h = min(patch_px, data.shape[0])
        w = min(patch_px, data.shape[1])
        out[:h, :w] = data[:h, :w]
        return out
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    sr = cfg["study_region"]
    raw = ROOT / cfg["paths"]["raw_dir"]
    processed = ROOT / cfg["paths"]["processed_dir"]

    spec = GridSpec(sr["lat_min"], sr["lat_max"], sr["lon_min"], sr["lon_max"],
                    sr["grid_resolution_km"])
    grid_df = load_or_build_grid(processed, spec)
    n = len(grid_df)

    tif = raw / "satellite" / "thermal_anomaly.tif"
    if not tif.exists():
        # Write a tiny sentinel — the Dataset detects this shape and emits
        # zero patches on-the-fly instead of materialising 3.8 GB of zeros.
        print(f"[skip] {tif} missing; writing sentinel (use_thermal will be ignored at load time)")
        np.save(processed / "thermal_patches.npy",
                np.zeros((1, 1, 1, 1), dtype=np.float32))
        masks_path = processed / "modality_masks.npy"
        masks = np.load(masks_path) if masks_path.exists() \
            else np.zeros((n, 4), dtype=bool)
        masks[:, 2] = False
        np.save(masks_path, masks)
        return

    import rasterio
    print(f"Sampling {tif} at {n} grid cells ...")
    patches = np.zeros((n, 1, PATCH, PATCH), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)

    with rasterio.open(tif) as ds:
        lats = grid_df["lat"].values
        lons = grid_df["lon"].values
        for i in tqdm(range(n)):
            patch = _read_window(ds, float(lons[i]), float(lats[i]), PATCH)
            valid_frac = np.isfinite(patch).mean()
            if valid_frac > 0.5:
                mask[i] = True
                patch = np.nan_to_num(patch, nan=np.nanmean(patch))
                patches[i, 0] = patch

    # Z-score using only valid patches
    if mask.any():
        mu = float(patches[mask].mean())
        sd = float(patches[mask].std()) or 1.0
        patches = (patches - mu) / sd
    else:
        mu, sd = 0.0, 1.0

    np.save(processed / "thermal_patches.npy", patches.astype(np.float32))
    print(f"Wrote thermal_patches.npy  shape={patches.shape}  coverage={mask.mean():.3f}")

    masks_path = processed / "modality_masks.npy"
    masks = np.load(masks_path) if masks_path.exists() else np.zeros((n, 4), dtype=bool)
    masks[:, 2] = mask
    np.save(masks_path, masks)

    stats_path = processed / "normalization_stats.json"
    stats = json.load(open(stats_path)) if stats_path.exists() else {}
    stats["thermal"] = {"mean": mu, "std": sd}
    json.dump(stats, open(stats_path, "w"), indent=2)


if __name__ == "__main__":
    main()
