"""Build the 6-channel geophysical stack and per-cell 32x32 patches.

Channels (Section 1.2.6 of CLAUDE.md):
    0 Bouguer gravity            (USGS GeoTIFF)
    1 Isostatic residual gravity (USGS GeoTIFF)
    2 Magnetic anomaly           (USGS / NOAA GeoTIFF)
    3 Heat flow (interpolated)   (SMU CSV)
    4 Temperature at 3.5 km      (computed from heat flow, 1D conduction)
    5 Temperature at 6.5 km      (computed from heat flow, 1D conduction)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import griddata
from tqdm import tqdm

from src.data.grid import GridSpec, load_or_build_grid

ROOT = Path(__file__).resolve().parents[2]


def _read_geotiff_to_grid(tif_path: Path, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Sample a GeoTIFF at the cell-center lat/lon. Returns NaN where missing."""
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.warp import transform as rio_transform
    except ImportError:
        raise SystemExit("rasterio is required for geophysics processing")

    if not tif_path.exists():
        return np.full(lats.shape, np.nan, dtype=np.float32)

    with rasterio.open(tif_path) as ds:
        # Some USGS GeoTIFFs ship without a CRS tag. By convention these
        # are geographic WGS84 (file names ending in _geog.tif).
        src_crs = ds.crs if ds.crs is not None else CRS.from_epsg(4326)
        if src_crs.to_string() == "EPSG:4326":
            coords = list(zip(lons.tolist(), lats.tolist()))
        else:
            xs, ys = rio_transform("EPSG:4326", src_crs, lons.tolist(), lats.tolist())
            coords = list(zip(xs, ys))
        samples = np.fromiter(
            (v[0] for v in ds.sample(coords)),
            dtype=np.float32,
            count=len(coords),
        )
        nodata = ds.nodata
        if nodata is not None:
            samples = np.where(samples == nodata, np.nan, samples)
        return samples


def _interp_heatflow(smu_dir: Path, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Linearly interpolate SMU heat-flow point data onto the cell grid."""
    if not smu_dir.exists():
        return np.full(lats.shape, np.nan, dtype=np.float32)

    frames = []
    for csv in smu_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv, low_memory=False)
        except Exception:
            continue
        df.columns = [c.lower() for c in df.columns]
        lat_col = next((c for c in df.columns if "lat" in c), None)
        lon_col = next((c for c in df.columns if "lon" in c), None)
        # Prefer the *site* heat-flow column; fall back to anything 'heat' or 'hf'.
        hf_col = next((c for c in df.columns if "siteheatflow" in c.replace("_", "")), None)
        if hf_col is None:
            hf_col = next((c for c in df.columns
                           if "heat" in c and "type" not in c and "unit" not in c
                           and "qual" not in c and "gen" not in c), None)
        if hf_col is None:
            hf_col = next((c for c in df.columns if "hf" == c), None)
        if not lat_col or not lon_col or not hf_col:
            continue
        sub = pd.DataFrame({
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
            "hf": pd.to_numeric(df[hf_col], errors="coerce"),
        }).dropna()
        frames.append(sub)

    if not frames:
        return np.full(lats.shape, np.nan, dtype=np.float32)

    pts = pd.concat(frames, ignore_index=True)
    # SMU uses -9999 / -99999 as "missing" sentinels — strip them out
    pts = pts[(pts.hf > 10) & (pts.hf < 500)]
    points = pts[["lon", "lat"]].values.astype(np.float64)
    values = pts.hf.values.astype(np.float64)
    grid_pts = np.column_stack([lons, lats])
    hf = griddata(points, values, grid_pts, method="linear")
    return hf.astype(np.float32)


def _temperature_at_depth(heat_flow: np.ndarray, z_km: float,
                          K: float = 2.5, A0: float = 2.5e-6,
                          T_surface: float = 15.0) -> np.ndarray:
    """1-D conductive temperature at depth (Blackwell & Richards, 2004).

        T(z) = T_s + (Q0/K) * z - (A0 * z^2) / (2 * K)

    Heat flow in mW/m^2 -> W/m^2 by *1e-3; depth in m.
    """
    z_m = z_km * 1000.0
    q0 = heat_flow * 1e-3  # W/m^2
    T = T_surface + (q0 / K) * z_m - (A0 * z_m ** 2) / (2 * K)
    return T.astype(np.float32)


def _zscore(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x)) or 1.0
    z = (x - mu) / sd
    return z.astype(np.float32), mu, sd


def _extract_patches(grid_df: pd.DataFrame, channels: np.ndarray,
                     patch: int = 32) -> np.ndarray:
    """Reshape channel arrays into per-cell patches using reflection padding."""
    n_rows = int(grid_df["row"].max()) + 1
    n_cols = int(grid_df["col"].max()) + 1
    n_ch = channels.shape[0]

    img = np.full((n_ch, n_rows, n_cols), np.nan, dtype=np.float32)
    img[:, grid_df["row"].values, grid_df["col"].values] = channels

    pad = patch // 2
    padded = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")

    n_cells = len(grid_df)
    patches = np.zeros((n_cells, n_ch, patch, patch), dtype=np.float32)
    rows = grid_df["row"].values + pad
    cols = grid_df["col"].values + pad
    for i in tqdm(range(n_cells), desc="geophys patches"):
        r, c = rows[i], cols[i]
        patches[i] = padded[:, r - pad:r + pad, c - pad:c + pad]
    return patches


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
    lats = grid_df["lat"].values.astype(np.float64)
    lons = grid_df["lon"].values.astype(np.float64)

    print("Sampling Bouguer gravity ...")
    bouguer = _read_geotiff_to_grid(raw / "geophysics" / "bouguer_gravity.tif", lats, lons)
    print("Sampling isostatic gravity ...")
    isostatic = _read_geotiff_to_grid(raw / "geophysics" / "isostatic_gravity.tif", lats, lons)
    print("Sampling magnetic anomaly ...")
    magnetic = _read_geotiff_to_grid(raw / "geophysics" / "magnetic_anomaly.tif", lats, lons)
    print("Interpolating SMU heat flow ...")
    hf = _interp_heatflow(raw / "geophysics" / "smu_heatflow", lats, lons)

    # Channels 4-5 are derived from heat flow (channel 3) by a 1-D conductive
    # model. This assumes pure conduction and uniform radiogenic heat
    # production — assumptions that *break down* in hydrothermally active
    # areas (i.e. our positive class). We keep them because they linearise
    # the heat-flow signal at depth, but ablation row A* in the paper
    # removes channels 3-5 to confirm that the model does not rely on this
    # circularity to discriminate positives.
    t35 = _temperature_at_depth(hf, z_km=3.5)
    t65 = _temperature_at_depth(hf, z_km=6.5)

    channels = np.stack([bouguer, isostatic, magnetic, hf, t35, t65], axis=0)

    has_heatflow = ~np.isnan(hf)
    fill_means = {}
    z_channels = []
    for i, ch in enumerate(channels):
        # impute NaNs with channel mean (after masking they don't contribute)
        mu = float(np.nanmean(ch)) if np.isfinite(ch).any() else 0.0
        ch_filled = np.where(np.isnan(ch), mu, ch)
        z, m, s = _zscore(ch_filled)
        fill_means[f"ch{i}"] = {"mean": m, "std": s}
        z_channels.append(z)
    z_channels = np.stack(z_channels, axis=0)

    patches = _extract_patches(grid_df, z_channels, patch=32)
    mask = has_heatflow  # primary availability gate; gravity/mag are nearly 100% covered

    out_patches = processed / "geophysics_patches.npy"
    np.save(out_patches, patches)
    print(f"Wrote {out_patches}  shape={patches.shape}")

    masks_path = processed / "modality_masks.npy"
    if masks_path.exists():
        masks = np.load(masks_path)
    else:
        masks = np.zeros((len(grid_df), 4), dtype=bool)
    masks[:, 0] = mask
    np.save(masks_path, masks)

    stats_path = processed / "normalization_stats.json"
    stats = json.load(open(stats_path)) if stats_path.exists() else {}
    stats["geophysics"] = fill_means
    json.dump(stats, open(stats_path, "w"), indent=2)
    print("done.")


if __name__ == "__main__":
    main()
