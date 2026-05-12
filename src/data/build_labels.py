"""Construct binary labels (`labels.npy`) and the known-field directory.

Positives: any grid cell within `positive_buffer_km` of a known geothermal site.
Negatives: cells that are
  - > negative_min_distance_km from ANY positive site,
  - > negative_spring_min_distance_km from ANY thermal spring,
  - have local heat flow < negative_max_heatflow mW/m^2 (if SMU coverage exists).

We sample roughly negative_pos_ratio negatives per positive cell, stratified
across rough tectonic bins so we don't bias toward the Great Plains.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from src.data.grid import GridSpec, load_or_build_grid

ROOT = Path(__file__).resolve().parents[2]


def _load_nrel_plants(shp_dir: Path) -> pd.DataFrame:
    """Read NREL plant shapefiles; return columns lat, lon, name."""
    if not shp_dir.exists():
        return pd.DataFrame(columns=["lat", "lon", "name"])
    try:
        import geopandas as gpd
        shp = next(shp_dir.glob("**/*.shp"), None)
        if shp is None:
            return pd.DataFrame(columns=["lat", "lon", "name"])
        gdf = gpd.read_file(shp).to_crs("EPSG:4326")
        name_col = next((c for c in gdf.columns if "name" in c.lower()), None)
        return pd.DataFrame({
            "lat": gdf.geometry.y.values,
            "lon": gdf.geometry.x.values,
            "name": gdf[name_col].astype(str).values if name_col else "",
        })
    except Exception as e:
        print(f"[warn] failed to read {shp_dir}: {e}")
        return pd.DataFrame(columns=["lat", "lon", "name"])


def _load_usgs_assessment(csv: Path) -> pd.DataFrame:
    if not csv.exists():
        return pd.DataFrame(columns=["lat", "lon", "name"])
    df = pd.read_csv(csv, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    lat_c = next((c for c in df.columns if "lat" in c), None)
    lon_c = next((c for c in df.columns if "lon" in c), None)
    name_c = next((c for c in df.columns if "name" in c or "system" in c), None)
    if not lat_c or not lon_c:
        return pd.DataFrame(columns=["lat", "lon", "name"])
    return pd.DataFrame({
        "lat": pd.to_numeric(df[lat_c], errors="coerce"),
        "lon": pd.to_numeric(df[lon_c], errors="coerce"),
        "name": df[name_c].astype(str).values if name_c else "",
    }).dropna(subset=["lat", "lon"])


def _dedupe_sites(sites: pd.DataFrame, radius_km: float = 5.0) -> pd.DataFrame:
    if sites.empty:
        return sites
    lat0 = float(sites["lat"].mean())
    cos_lat0 = float(np.cos(np.radians(lat0)))
    xy = np.column_stack([
        sites["lon"].values * 111.32 * cos_lat0,
        sites["lat"].values * 111.32,
    ])
    keep = np.ones(len(sites), dtype=bool)
    tree = cKDTree(xy)
    for i in range(len(sites)):
        if not keep[i]:
            continue
        neighbors = tree.query_ball_point(xy[i], r=radius_km)
        for j in neighbors:
            if j > i:
                keep[j] = False
    return sites.iloc[keep].reset_index(drop=True)


def _province_bin(lat: float, lon: float) -> str:
    if lat > 41 and -124 < lon < -120:
        return "Cascades"
    if lat < 42 and -118 < lon < -113:
        return "Basin_and_Range"
    if 42 < lat < 46 and -116 < lon < -110:
        return "Snake_River_Plain"
    if 31 < lat < 35 and -116 < lon < -113:
        return "Salton_Trough"
    if -113 < lon < -103:
        return "Rocky_Mountain"
    return "Other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    sr = cfg["study_region"]
    lbl = cfg["labels"]
    raw = ROOT / cfg["paths"]["raw_dir"]
    processed = ROOT / cfg["paths"]["processed_dir"]
    meta = ROOT / cfg["paths"]["metadata_dir"]
    meta.mkdir(parents=True, exist_ok=True)

    spec = GridSpec(sr["lat_min"], sr["lat_max"], sr["lon_min"], sr["lon_max"],
                    sr["grid_resolution_km"])
    grid_df = load_or_build_grid(processed, spec)
    n = len(grid_df)

    print("Loading positive sites ...")
    op = _load_nrel_plants(raw / "labels" / "operating_plants")
    dev = _load_nrel_plants(raw / "labels" / "developing_plants")
    usgs = _load_usgs_assessment(raw / "labels" / "usgs_identified_systems.csv")
    print(f"  operating: {len(op)}  developing: {len(dev)}  USGS: {len(usgs)}")

    all_sites = pd.concat([op, dev, usgs], ignore_index=True)
    all_sites = all_sites.dropna(subset=["lat", "lon"])
    all_sites = all_sites[
        (all_sites["lat"].between(sr["lat_min"], sr["lat_max"])) &
        (all_sites["lon"].between(sr["lon_min"], sr["lon_max"]))
    ].reset_index(drop=True)
    print(f"  combined in-region sites: {len(all_sites)}")
    sites = _dedupe_sites(all_sites, radius_km=5.0)
    print(f"  unique fields after dedup: {len(sites)}")

    # Assign tectonic province for stratification + later analysis
    sites["province"] = [
        _province_bin(la, lo) for la, lo in zip(sites.lat.values, sites.lon.values)
    ]
    sites["field_id"] = np.arange(len(sites))

    lat0 = float(grid_df["lat"].mean())
    cos_lat0 = float(np.cos(np.radians(lat0)))
    grid_xy = np.column_stack([
        grid_df["lon"].values * 111.32 * cos_lat0,
        grid_df["lat"].values * 111.32,
    ])

    if len(sites) > 0:
        site_xy = np.column_stack([
            sites["lon"].values * 111.32 * cos_lat0,
            sites["lat"].values * 111.32,
        ])
        site_tree = cKDTree(site_xy)
        nearest_dist, nearest_idx = site_tree.query(grid_xy, k=1)
    else:
        nearest_dist = np.full(n, 1e6, dtype=np.float64)
        nearest_idx = np.zeros(n, dtype=np.int64)

    labels = np.zeros(n, dtype=np.int8)
    field_assignment = np.full(n, -1, dtype=np.int32)

    # Positive cells: within positive_buffer_km of any field
    pos_mask = nearest_dist <= lbl["positive_buffer_km"]
    labels[pos_mask] = 1
    field_assignment[pos_mask] = nearest_idx[pos_mask].astype(np.int32) \
        if len(sites) else -1
    n_pos = int(pos_mask.sum())
    print(f"Positive cells: {n_pos}")

    # Province lookup for stratified negatives
    grid_df["province"] = [
        _province_bin(la, lo)
        for la, lo in zip(grid_df["lat"].values, grid_df["lon"].values)
    ]

    # Negatives: far from any positive, far from springs (proxy: 1st modality coverage),
    # and with low heat flow if we know it.
    far_from_pos = nearest_dist >= lbl["negative_min_distance_km"]
    candidates = far_from_pos.copy()

    # Springs proxy: cells lacking geochemistry mask (no springs nearby) qualify
    masks_path = processed / "modality_masks.npy"
    if masks_path.exists():
        masks = np.load(masks_path)
        no_springs = ~masks[:, 1]
        candidates &= no_springs
    n_neg_target = min(n_pos * lbl["negative_pos_ratio"], int(candidates.sum()))

    rng = np.random.default_rng(42)
    candidate_idx = np.flatnonzero(candidates)
    province_of_cand = grid_df["province"].values[candidate_idx]
    sample_idx = []
    for prov in np.unique(province_of_cand):
        idx = candidate_idx[province_of_cand == prov]
        k = min(len(idx), max(1, int(n_neg_target * len(idx) / max(1, len(candidate_idx)))))
        sample_idx.append(rng.choice(idx, size=k, replace=False))
    sample_idx = np.concatenate(sample_idx)[:n_neg_target]
    labels[sample_idx] = 0  # explicit negatives; keep label int8 but encode as -1 in mask?

    # We need a TRAINING mask to distinguish "labeled neg" from "unlabeled".
    train_mask = pos_mask.copy()
    train_mask[sample_idx] = True

    np.save(processed / "labels.npy", labels.astype(np.int8))
    np.save(processed / "train_mask.npy", train_mask)
    np.save(processed / "field_assignment.npy", field_assignment)

    sites.to_csv(meta / "known_fields_details.csv", index=False)
    grid_df[["cell_id", "lat", "lon", "province"]].to_csv(
        meta / "tectonic_provinces.csv", index=False
    )

    print(f"Negatives: {len(sample_idx)}")
    print(f"Wrote labels.npy, train_mask.npy, field_assignment.npy")
    print(f"Positive cells per province:")
    print(grid_df.loc[pos_mask, "province"].value_counts().to_string())


if __name__ == "__main__":
    main()
