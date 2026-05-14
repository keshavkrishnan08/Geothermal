"""Compute the 11-dim geological structure feature vector per grid cell.

Features (DATA.md (geology section)):
    0  fault_density_25km            sum of fault length (km) within 25 km
    1  nearest_fault_distance_km
    2  max_slip_rate_25km            (mm/yr)
    3  fault_intersection_density_10km
    4  nearest_volcano_distance_km   (Holocene)
    5  n_volcanoes_100km
    6  elevation_m                   (SRTM, cell-center sample)
    7  rock_volcanic                 (binary, from SGMC if available)
    8  rock_intrusive                (binary)
    9  rock_sedimentary              (binary)
    10 rock_quaternary               (binary)

If SGMC lithology is unavailable, features 7-10 stay zero (paper notes this).
The structural mask is always True (faults + volcanoes are global resources).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from tqdm import tqdm

from src.data.grid import GridSpec, load_or_build_grid

ROOT = Path(__file__).resolve().parents[2]


def _load_faults(raw: Path):
    """Returns list of (line_xy_metric_km, slip_rate_mm_per_yr) for active faults."""
    import geopandas as gpd
    fault_dir = raw / "geology" / "qfaults"
    shp = next(fault_dir.glob("**/*.shp"), None) if fault_dir.exists() else None
    if shp is None:
        return [], gpd.GeoDataFrame()
    gdf = gpd.read_file(shp)
    gdf = gdf.to_crs("EPSG:4326")
    # USGS Qfaults stores slip rate as a categorical string (e.g. "Less than
    # 0.2 mm/yr", "Between 1.0 and 5.0 mm/yr"). Parse to a numeric mid-range
    # so the model gets a useful continuous signal.
    slip_col = next((c for c in gdf.columns if c.lower() == "slip_rate"
                     or "slip_rate" in c.lower()), None)
    if slip_col is None:
        slip_col = next((c for c in gdf.columns if "slip" in c.lower()), None)

    def _parse_slip(s):
        if not isinstance(s, str):
            try:
                return float(s)
            except (TypeError, ValueError):
                return 0.0
        sl = s.lower()
        if "less than 0.2" in sl: return 0.1
        if "between 0.2 and 1.0" in sl: return 0.6
        if "between 1.0 and 5.0" in sl: return 3.0
        if "greater than 5.0" in sl: return 10.0
        if "unspecified" in sl or "insufficient" in sl: return 0.0
        # Try numeric extraction "0.2 +/- 0.1 mm/yr"
        import re
        m = re.search(r"(-?\d+(?:\.\d+)?)", s)
        return float(m.group(1)) if m else 0.0

    if slip_col:
        gdf["slip_rate"] = gdf[slip_col].apply(_parse_slip)
    else:
        gdf["slip_rate"] = 0.0
    return gdf


def _load_volcanoes(raw: Path) -> pd.DataFrame:
    path = raw / "geology" / "holocene_volcanoes.csv"
    if not path.exists():
        return pd.DataFrame(columns=["lat", "lon"])
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    lat_c = next((c for c in df.columns if "lat" in c), None)
    lon_c = next((c for c in df.columns if "lon" in c), None)
    if not lat_c or not lon_c:
        return pd.DataFrame(columns=["lat", "lon"])
    return pd.DataFrame({
        "lat": pd.to_numeric(df[lat_c], errors="coerce"),
        "lon": pd.to_numeric(df[lon_c], errors="coerce"),
    }).dropna()


def _sample_sgmc_lithology(raw: Path, lats: np.ndarray, lons: np.ndarray):
    """Spatial-join lithology categories from USGS SGMC geodatabase to grid cells.

    Returns a dict with four boolean arrays {volcanic, intrusive, sedimentary,
    quaternary} of length len(lats), or None if SGMC isn't downloaded.
    Falls back gracefully if the geodatabase is missing or malformed.
    """
    sgmc_dir = raw / "geology" / "sgmc"
    if not sgmc_dir.exists():
        print("  [skip] SGMC geodatabase not present; rock-type features remain zero")
        return None

    import geopandas as gpd
    from shapely.geometry import Point

    # Find the geodatabase or shapefile
    gdb = next(sgmc_dir.glob("**/*.gdb"), None)
    shp = next(sgmc_dir.glob("**/SGMC_Geology.shp"), None) or next(sgmc_dir.glob("**/*.shp"), None)
    src = gdb or shp
    if src is None:
        print("  [skip] no .gdb or .shp found under data/raw/geology/sgmc/")
        return None

    print(f"  reading {src.name} ...")
    try:
        gdf = gpd.read_file(src, layer="SGMC_Geology") if gdb else gpd.read_file(src)
    except Exception as e:
        print(f"  [warn] failed to read SGMC: {e}")
        return None
    gdf = gdf.to_crs("EPSG:4326")

    # Restrict to western US bbox to keep the spatial join cheap
    gdf = gdf.cx[-125:-103, 31:49]
    print(f"  SGMC polygons in western US: {len(gdf):,}")
    if len(gdf) == 0:
        return None

    # SGMC_Geology has a GENERALIZED_LITH column directly — preferred.
    lith_col = None
    for cand in ("GENERALIZED_LITH", "GENERALIZ", "MAJOR1", "ROCKTYPE1", "LITH1", "LITHOLOGY", "ROCKTYPE"):
        if cand in gdf.columns:
            lith_col = cand
            break
    if lith_col is None:
        # SGMC uses UNIT_LINK to join to lithology table
        link_col = next((c for c in gdf.columns if "UNIT_LINK" in c.upper()), None)
        lith_csv = sgmc_dir / "USGS_SGMC_Tables_CSV" / "SGMC_Lithology.csv"
        if link_col and lith_csv.exists():
            lith = pd.read_csv(lith_csv, low_memory=False)
            gdf = gdf.merge(lith[["UNIT_LINK", "LITH1"]], left_on=link_col,
                            right_on="UNIT_LINK", how="left")
            lith_col = "LITH1"
    if lith_col is None:
        print("  [warn] could not find lithology column; rock-type features stay zero")
        return None

    gdf["_lith"] = gdf[lith_col].astype(str).str.lower().fillna("")

    # Spatial join cells -> polygons. Done in chunks so we can show progress
    # and so memory stays bounded. R-tree is built once on the (large) gdf.
    sgmc = gdf[["geometry", "_lith"]].reset_index(drop=True)
    print(f"  spatial-join {len(lats):,} cells against {len(sgmc):,} polygons ...")
    chunk = 5000
    out = np.full(len(lats), "", dtype=object)
    from tqdm import tqdm as _tqdm
    for start in _tqdm(range(0, len(lats), chunk), desc="sgmc sjoin"):
        end = min(start + chunk, len(lats))
        sub = gpd.GeoDataFrame(
            {"_idx": np.arange(start, end)},
            geometry=[Point(lo, la) for la, lo in zip(lats[start:end], lons[start:end])],
            crs="EPSG:4326",
        )
        j = gpd.sjoin(sub, sgmc, how="left", predicate="within")
        j = j.loc[~j["_idx"].duplicated(keep="first")]
        out[j["_idx"].values] = j["_lith"].fillna("").values
    lith_per_cell = out

    def _contains(needles):
        return np.array([any(n in s for n in needles) for s in lith_per_cell])

    # Match against the SGMC GENERALIZED_LITH categories — these are
    # the actual category strings, not free-text descriptions.
    return {
        "volcanic":    _contains(("igneous, volcanic", "metamorphic, volcanic", "basalt", "andesite", "rhyolite")).astype(np.float32),
        "intrusive":   _contains(("igneous, intrusive", "metamorphic, intrusive", "granite", "diorite", "gabbro")).astype(np.float32),
        "sedimentary": _contains(("sedimentary", "sandstone", "shale", "limestone", "conglomerate", "siltstone")).astype(np.float32),
        "quaternary":  _contains(("unconsolidated", "alluvi", "glacial", "till", "loess", "colluvi")).astype(np.float32),
    }


def _sample_srtm(raw: Path, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    tif = raw / "geology" / "srtm_elevation.tif"
    if not tif.exists():
        return np.zeros(len(lats), dtype=np.float32)
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform as rio_transform
    with rasterio.open(tif) as ds:
        src_crs = ds.crs if ds.crs is not None else CRS.from_epsg(4326)
        if src_crs.to_string() == "EPSG:4326":
            coords = list(zip(lons.tolist(), lats.tolist()))
        else:
            xs, ys = rio_transform("EPSG:4326", src_crs, lons.tolist(), lats.tolist())
            coords = list(zip(xs, ys))
        vals = np.fromiter((v[0] for v in ds.sample(coords)), dtype=np.float32, count=len(coords))
        if ds.nodata is not None:
            vals = np.where(vals == ds.nodata, 0.0, vals)
    return vals


def _fault_intersections(fault_gdf, cos_lat0, lat0):
    """Approximate fault intersections by fault-endpoint clustering.

    Exact pairwise shapely intersection on ~10k fault segments takes hours.
    We approximate: collect all fault line endpoints, find endpoints that
    cluster within 1 km of an endpoint from a *different* fault. This
    captures the geologically meaningful "fault-tip / intersection" zones
    that control geothermal circulation, without quadratic geometric ops.
    """
    if len(fault_gdf) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    endpoints = []
    owners = []
    for fid, geom in enumerate(fault_gdf.geometry.values):
        if geom is None or geom.is_empty:
            continue
        if hasattr(geom, "geoms"):
            for sub in geom.geoms:
                coords = list(sub.coords)
                if len(coords) >= 2:
                    endpoints.append(coords[0]); owners.append(fid)
                    endpoints.append(coords[-1]); owners.append(fid)
        else:
            coords = list(geom.coords)
            if len(coords) >= 2:
                endpoints.append(coords[0]); owners.append(fid)
                endpoints.append(coords[-1]); owners.append(fid)
    if not endpoints:
        return np.zeros((0, 2), dtype=np.float64)

    pts = np.array(endpoints)
    xy = np.column_stack([pts[:, 0] * 111.32 * cos_lat0, pts[:, 1] * 111.32])
    owners = np.array(owners)

    # Find endpoint pairs within 1 km belonging to different faults — these
    # are our intersection proxies.
    tree = cKDTree(xy)
    pairs = tree.query_pairs(r=1.0, output_type="ndarray")
    if len(pairs) == 0:
        return xy[:0]
    diff_owner = owners[pairs[:, 0]] != owners[pairs[:, 1]]
    pairs = pairs[diff_owner]
    if len(pairs) == 0:
        return xy[:0]
    # Use the midpoint of each cross-fault pair as the intersection location.
    mids = 0.5 * (xy[pairs[:, 0]] + xy[pairs[:, 1]])
    return mids


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
    lats = grid_df["lat"].values
    lons = grid_df["lon"].values
    lat0 = float(lats.mean())
    cos_lat0 = float(np.cos(np.radians(lat0)))

    grid_xy = np.column_stack([
        lons * 111.32 * cos_lat0,
        lats * 111.32,
    ])

    print("Loading Quaternary faults ...")
    faults = _load_faults(raw)
    print(f"  faults: {len(faults)}")
    print("Loading Holocene volcanoes ...")
    volcanoes = _load_volcanoes(raw)
    print(f"  volcanoes: {len(volcanoes)}")
    print("Sampling SRTM elevation ...")
    elev = _sample_srtm(raw, lats, lons)

    feats = np.zeros((n, 11), dtype=np.float32)

    # ---- Volcanoes ----------------------------------------------------------
    if len(volcanoes) > 0:
        v_xy = np.column_stack([
            volcanoes["lon"].values * 111.32 * cos_lat0,
            volcanoes["lat"].values * 111.32,
        ])
        v_tree = cKDTree(v_xy)
        dists, _ = v_tree.query(grid_xy, k=1)
        feats[:, 4] = dists
        for i in tqdm(range(n), desc="volcano counts"):
            idx = v_tree.query_ball_point(grid_xy[i], r=100.0)
            feats[i, 5] = len(idx)
    else:
        feats[:, 4] = 1000.0

    # ---- Faults --------------------------------------------------------------
    if len(faults) > 0:
        # Sample points along each fault line at ~2 km spacing
        fault_pts = []
        slip_pts = []
        for geom, slip in zip(faults.geometry.values, faults["slip_rate"].values):
            if geom is None or geom.is_empty:
                continue
            length = geom.length  # in degrees; convert later
            step_deg = 2.0 / 111.32  # ~2 km step
            n_steps = max(2, int(length / step_deg))
            for t in np.linspace(0, 1, n_steps):
                pt = geom.interpolate(t, normalized=True)
                fault_pts.append((pt.x, pt.y))
                slip_pts.append(float(slip))
        fault_pts = np.array(fault_pts) if fault_pts else np.zeros((0, 2))
        slip_pts = np.array(slip_pts) if slip_pts else np.zeros(0)
        if len(fault_pts) > 0:
            f_xy = np.column_stack([
                fault_pts[:, 0] * 111.32 * cos_lat0,
                fault_pts[:, 1] * 111.32,
            ])
            f_tree = cKDTree(f_xy)
            dists, _ = f_tree.query(grid_xy, k=1)
            feats[:, 1] = dists

            slip_pts = np.nan_to_num(slip_pts, nan=0.0)
            for i in tqdm(range(n), desc="fault density"):
                idx = f_tree.query_ball_point(grid_xy[i], r=25.0)
                feats[i, 0] = len(idx) * 2.0  # ~2 km per sample point
                if idx:
                    v = float(np.nanmax(slip_pts[idx]))
                    feats[i, 2] = 0.0 if not np.isfinite(v) else v

            # Intersection proxy
            xs_int = _fault_intersections(faults, cos_lat0, lat0)
            if len(xs_int) > 0:
                i_tree = cKDTree(xs_int)
                for i in tqdm(range(n), desc="fault intersections"):
                    idx = i_tree.query_ball_point(grid_xy[i], r=10.0)
                    feats[i, 3] = len(idx)
        else:
            feats[:, 1] = 1000.0
    else:
        feats[:, 1] = 1000.0

    # ---- Elevation ----------------------------------------------------------
    feats[:, 6] = elev

    # ---- Lithology (binary rock-type from SGMC) -----------------------------
    print("Sampling SGMC lithology ...")
    rock_flags = _sample_sgmc_lithology(raw, lats, lons)
    if rock_flags is not None:
        feats[:, 7] = rock_flags["volcanic"]
        feats[:, 8] = rock_flags["intrusive"]
        feats[:, 9] = rock_flags["sedimentary"]
        feats[:, 10] = rock_flags["quaternary"]
        coverage = np.any(np.column_stack([rock_flags[k] for k in
                          ("volcanic","intrusive","sedimentary","quaternary")]), axis=1).mean()
        print(f"  SGMC coverage: {coverage:.3f}")

    # ---- Z-score continuous (0-6), leave binary (7-10) as 0/1 ---------------
    stats = {}
    z = feats.copy()
    for c in range(7):
        mu = float(feats[:, c].mean())
        sd = float(feats[:, c].std()) or 1.0
        z[:, c] = (feats[:, c] - mu) / sd
        stats[f"f{c}"] = {"mean": mu, "std": sd}

    np.save(processed / "geology_features.npy", z.astype(np.float32))
    print(f"Wrote geology_features.npy  shape={z.shape}")

    masks_path = processed / "modality_masks.npy"
    masks = np.load(masks_path) if masks_path.exists() else np.zeros((n, 4), dtype=bool)
    masks[:, 3] = True  # geology is always available
    np.save(masks_path, masks)

    stats_path = processed / "normalization_stats.json"
    s = json.load(open(stats_path)) if stats_path.exists() else {}
    s["geology"] = stats
    json.dump(s, open(stats_path, "w"), indent=2)


if __name__ == "__main__":
    main()
