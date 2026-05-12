"""Build the 9-dim geochemistry feature vector per grid cell.

Features (Section 1.3.4 of CLAUDE.md):
    0 n_springs                       count of thermal springs within 50 km
    1 max_spring_temperature_C       (NOAA)
    2 mean_chalcedony_T_C            silica geothermometer
    3 mean_NaK_T_C                   Na-K geothermometer
    4 max_BHT_C                      (SMU / NGDS bottom-hole temps)
    5 mean_thermal_gradient_C_per_km (SMU)
    6 n_chemistry_samples            (GEOTHERM coverage indicator)
    7 max_Li_ppm                     (deep fluid circulation)
    8 mean_Cl_ppm                    (deep brine indicator)

Geothermometer formulas: Fournier (1977), Giggenbach (1988). Concentrations
are expected in mg/L (=ppm for dilute waters).
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty")
from scipy.spatial import cKDTree
from tqdm import tqdm

from src.data.grid import GridSpec, load_or_build_grid, haversine_km

ROOT = Path(__file__).resolve().parents[2]


def _silica_chalcedony(sio2_ppm: pd.Series) -> pd.Series:
    """T = 1032 / (4.69 - log10(SiO2)) - 273.15, valid for T < 180C."""
    x = pd.to_numeric(sio2_ppm, errors="coerce")
    log = np.log10(x.where(x > 0))
    T = 1032.0 / (4.69 - log) - 273.15
    return T.where((T > 20) & (T < 350))


def _nak(na_ppm: pd.Series, k_ppm: pd.Series) -> pd.Series:
    """T = 1217 / (log10(Na/K) + 1.483) - 273.15."""
    na = pd.to_numeric(na_ppm, errors="coerce")
    k = pd.to_numeric(k_ppm, errors="coerce")
    ratio = (na / k).where((na > 0) & (k > 0))
    T = 1217.0 / (np.log10(ratio) + 1.483) - 273.15
    return T.where((T > 20) & (T < 400))


_DMS_RE = re.compile(r"^\s*(-?\d+)[-\s](\d+(?:\.\d+)?)\s*([NSEW]?)\s*$")


def _parse_dms(value) -> float:
    """Parse a coordinate value that may be decimal degrees OR DMS (e.g.
    '56-49.95 N', '135-22.25 W'). Returns decimal degrees or NaN."""
    if pd.isna(value):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        s = str(value).strip()
        m = _DMS_RE.match(s)
        if not m:
            return float("nan")
        deg, mn, hemi = m.groups()
        dec = float(deg) + float(mn) / 60.0
        if hemi in ("S", "W"):
            dec = -dec
        elif float(deg) < 0:
            dec = -abs(dec)
        return dec


def _read_geotherm(raw: Path) -> pd.DataFrame:
    path = raw / "geochemistry" / "geotherm.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]

    def col(*names):
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series(np.nan, index=df.index)

    lat_raw = col("lat", "latitude", "decimal_lat")
    lon_raw = col("lon", "longitude", "decimal_lon")
    out = pd.DataFrame({
        "lat": lat_raw.apply(_parse_dms),
        "lon": lon_raw.apply(_parse_dms),
        "sio2": pd.to_numeric(col("sio2", "silica"), errors="coerce"),
        "na": pd.to_numeric(col("na", "sodium"), errors="coerce"),
        "k": pd.to_numeric(col("k", "potassium"), errors="coerce"),
        "li": pd.to_numeric(col("li", "lithium"), errors="coerce"),
        "cl": pd.to_numeric(col("cl", "chloride"), errors="coerce"),
        "t_measured": pd.to_numeric(col("temp", "temperature", "t_c"), errors="coerce"),
    }).dropna(subset=["lat", "lon"])
    out["T_chalcedony"] = _silica_chalcedony(out["sio2"])
    out["T_NaK"] = _nak(out["na"], out["k"])
    return out


def _read_thermal_springs(raw: Path) -> pd.DataFrame:
    path = raw / "geochemistry" / "thermal_springs.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    lat_c = next((c for c in df.columns if "lat" in c), None)
    lon_c = next((c for c in df.columns if "lon" in c), None)
    temp_c = next((c for c in df.columns if "temp" in c or "tem" in c), None)
    if not lat_c or not lon_c:
        return pd.DataFrame()
    out = pd.DataFrame({
        "lat": pd.to_numeric(df[lat_c], errors="coerce"),
        "lon": pd.to_numeric(df[lon_c], errors="coerce"),
        "spring_T": pd.to_numeric(df[temp_c], errors="coerce") if temp_c else np.nan,
    }).dropna(subset=["lat", "lon"])
    return out


def _read_smu_bht(raw: Path) -> pd.DataFrame:
    smu_dir = raw / "geophysics" / "smu_heatflow"
    if not smu_dir.exists():
        return pd.DataFrame()
    frames = []
    for csv in smu_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv, low_memory=False)
        except Exception:
            continue
        df.columns = [c.lower() for c in df.columns]
        lat_c = next((c for c in df.columns if "lat" in c), None)
        lon_c = next((c for c in df.columns if "lon" in c), None)
        bht_c = next((c for c in df.columns if "bht" in c or "bottom" in c), None)
        grad_c = next((c for c in df.columns if "grad" in c), None)
        if not lat_c or not lon_c:
            continue
        sub = pd.DataFrame({
            "lat": pd.to_numeric(df[lat_c], errors="coerce"),
            "lon": pd.to_numeric(df[lon_c], errors="coerce"),
            "bht": pd.to_numeric(df[bht_c], errors="coerce") if bht_c else np.nan,
            "grad": pd.to_numeric(df[grad_c], errors="coerce") if grad_c else np.nan,
        })
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).dropna(subset=["lat", "lon"])


def _aggregate(grid_df: pd.DataFrame,
               points: pd.DataFrame, radius_km: float = 50.0) -> pd.DataFrame:
    """Aggregate point measurements within `radius_km` of each cell using a KDTree
    on equirectangular-projected coordinates (sufficient at 50 km scale).
    """
    if points.empty:
        return pd.DataFrame(index=grid_df.index)

    # Approximate metric coords (km) — good enough for radius queries at CONUS scale.
    lat0 = float(grid_df["lat"].mean())
    def to_xy(lat, lon):
        x = (np.asarray(lon, dtype=np.float64) * 111.32 * np.cos(np.radians(lat0)))
        y = (np.asarray(lat, dtype=np.float64) * 111.32)
        return np.column_stack([x, y])

    grid_xy = to_xy(grid_df["lat"].values, grid_df["lon"].values)
    pts_xy = to_xy(points["lat"].values, points["lon"].values)
    tree = cKDTree(pts_xy)
    return tree, grid_xy, pts_xy


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

    print(f"Loading geochemistry sources ...")
    chem = _read_geotherm(raw)
    springs = _read_thermal_springs(raw)
    bht = _read_smu_bht(raw)
    print(f"  geotherm rows: {len(chem)}  springs: {len(springs)}  smu/bht: {len(bht)}")

    feats = np.zeros((n, 9), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)

    radius = 50.0
    lat0 = float(grid_df["lat"].mean())
    cos_lat0 = float(np.cos(np.radians(lat0)))

    def make_tree(df: pd.DataFrame):
        if df.empty:
            return None
        xy = np.column_stack([
            df["lon"].values * 111.32 * cos_lat0,
            df["lat"].values * 111.32,
        ])
        return cKDTree(xy)

    grid_xy = np.column_stack([
        grid_df["lon"].values * 111.32 * cos_lat0,
        grid_df["lat"].values * 111.32,
    ])

    chem_tree = make_tree(chem)
    spring_tree = make_tree(springs)
    bht_tree = make_tree(bht)

    def _safe(v):
        # Replace NaN / inf with 0 — Python's `or 0.0` doesn't catch NaN.
        v = float(v)
        return 0.0 if not np.isfinite(v) else v

    for i in tqdm(range(n), desc="geochem aggregate"):
        x = grid_xy[i]
        if spring_tree is not None:
            idx = spring_tree.query_ball_point(x, r=radius)
            if idx:
                feats[i, 0] = len(idx)
                feats[i, 1] = _safe(np.nanmax(springs.iloc[idx]["spring_T"]))
                mask[i] = True
        if chem_tree is not None:
            idx = chem_tree.query_ball_point(x, r=radius)
            if idx:
                sub = chem.iloc[idx]
                feats[i, 2] = _safe(np.nanmean(sub["T_chalcedony"]))
                feats[i, 3] = _safe(np.nanmean(sub["T_NaK"]))
                feats[i, 6] = len(idx)
                feats[i, 7] = _safe(np.nanmax(sub["li"]))
                feats[i, 8] = _safe(np.nanmean(sub["cl"]))
                mask[i] = True
        if bht_tree is not None:
            idx = bht_tree.query_ball_point(x, r=radius)
            if idx:
                sub = bht.iloc[idx]
                feats[i, 4] = _safe(np.nanmax(sub["bht"]))
                feats[i, 5] = _safe(np.nanmean(sub["grad"]))
                mask[i] = True

    # log-transform highly skewed columns BEFORE z-score
    for col in (0, 6, 7, 8):
        feats[:, col] = np.log1p(np.clip(feats[:, col], 0, None))

    stats = {}
    z = np.zeros_like(feats)
    for c in range(feats.shape[1]):
        mu = float(feats[mask, c].mean()) if mask.any() else 0.0
        sd = float(feats[mask, c].std()) or 1.0
        z[:, c] = (feats[:, c] - mu) / sd
        stats[f"f{c}"] = {"mean": mu, "std": sd}

    np.save(processed / "geochemistry_features.npy", z.astype(np.float32))
    print(f"Wrote geochemistry_features.npy  shape={z.shape}  coverage={mask.mean():.3f}")

    masks_path = processed / "modality_masks.npy"
    if masks_path.exists():
        masks = np.load(masks_path)
    else:
        masks = np.zeros((n, 4), dtype=bool)
    masks[:, 1] = mask
    np.save(masks_path, masks)

    stats_path = processed / "normalization_stats.json"
    s = json.load(open(stats_path)) if stats_path.exists() else {}
    s["geochemistry"] = stats
    json.dump(s, open(stats_path, "w"), indent=2)


if __name__ == "__main__":
    main()
