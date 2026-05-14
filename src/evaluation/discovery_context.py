"""Per-cluster geological context for the 33 consensus discoveries.

For each discovery, computes:
  - tectonic province (already in the discoveries CSV)
  - distance to nearest known geothermal field (already there as dist_to_max_km)
  - distance to nearest Quaternary fault (this script)
  - distance to nearest documented active volcano / Quaternary igneous centre
  - inferred-system type label (binary-cycle, flash-plant, direct-use)
  - regional structural setting (Walker Lane, SRP hot-spot trace, etc.)

This is the geological table a Geothermics reviewer will want before
trusting the candidates. The output joins the discoveries CSV with the
geological annotation columns and is sorted by P50 MWe.

Output
------
outputs/results/discovery_geological_context.csv
outputs/results/discovery_geological_context.md  (formatted for the paper)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]


# Hand-curated structural-setting polygons (approximate bounding boxes).
# Used purely for labelling; not part of the model.
STRUCTURAL_SETTINGS = [
    # name, lat_min, lat_max, lon_min, lon_max
    ("Walker Lane",                 35.0, 41.5, -120.5, -117.0),
    ("Eastern Nevada extension",    38.0, 41.5, -116.5, -114.0),
    ("Eastern California shear",    33.0, 36.5, -118.5, -115.5),
    ("Snake River Plain hot-spot",  42.0, 44.5, -116.0, -111.5),
    ("Cascades arc",                40.5, 49.0, -123.5, -120.5),
    ("Imperial Valley",             32.0, 33.5, -116.5, -114.5),
    ("Yellowstone halo",            43.5, 45.5, -111.5, -109.5),
    ("Long Valley caldera",         37.5, 38.0, -119.2, -118.5),
    ("Coso volcanic field",         35.7, 36.3, -118.0, -117.5),
    ("Klamath/N Cascades",          40.8, 42.5, -121.8, -120.5),
]


def structural_label(lat, lon):
    for name, la_min, la_max, lo_min, lo_max in STRUCTURAL_SETTINGS:
        if la_min <= lat <= la_max and lo_min <= lon <= lo_max:
            return name
    return "other"


def temperature_class(T):
    if T >= 200:
        return "flash-plant grade (>=200 deg C)"
    if T >= 180:
        return "binary-cycle grade (180-200 deg C)"
    if T >= 130:
        return "binary low-T (130-180 deg C)"
    if T >= 90:
        return "district heating (90-130 deg C)"
    return "direct-use (<90 deg C)"


def load_quaternary_faults():
    """Return a 2-D numpy array of fault-trace centroids in km (EPSG:5070
    projected via the same flat-earth approx the rest of the pipeline uses)."""
    import shapefile
    shp = ROOT / "data/raw/geology/qfaults/SHP/Qfaults_US_Database.shp"
    if not shp.exists():
        print(f"[warn] Quaternary faults shapefile missing at {shp}")
        return None
    print(f"loading {shp.name} ...")
    sf = shapefile.Reader(str(shp))
    pts = []
    for s in sf.shapes():
        if s.shapeType in (3, 5, 13, 15):  # polyline / polygon (+ z)
            arr = np.asarray(s.points, dtype=np.float64)
            if len(arr):
                pts.append(arr.mean(axis=0))  # use centroid for distance
    if not pts:
        return None
    pts = np.array(pts)  # (N, 2) = (lon, lat)
    # Filter to western US for speed
    keep = (pts[:, 0] > -125) & (pts[:, 0] < -103) \
           & (pts[:, 1] > 31) & (pts[:, 1] < 49)
    pts = pts[keep]
    print(f"  {len(pts):,} western-US Q-fault centroids loaded")
    return pts


def nearest_distance_km(targets_lonlat, query_lonlat):
    """For each row in query_lonlat (Nx2), distance to nearest target."""
    lat0 = float(np.mean(targets_lonlat[:, 1]))
    cos0 = float(np.cos(np.radians(lat0)))
    txy = np.column_stack([targets_lonlat[:, 0] * 111.32 * cos0,
                            targets_lonlat[:, 1] * 111.32])
    qxy = np.column_stack([query_lonlat[:, 0] * 111.32 * cos0,
                            query_lonlat[:, 1] * 111.32])
    tree = cKDTree(txy)
    d, _ = tree.query(qxy, k=1)
    return d


def main():
    results = ROOT / "outputs/results"
    disc = pd.read_csv(results / "consensus_with_mwe_v2.csv")
    print(f"loaded {len(disc)} consensus discoveries")

    # Quaternary fault distance
    qf = load_quaternary_faults()
    if qf is not None:
        d_fault = nearest_distance_km(
            qf, disc[["lon", "lat"]].values
        )
        disc["dist_to_nearest_qfault_km"] = d_fault
    else:
        disc["dist_to_nearest_qfault_km"] = np.nan

    # Volcanic / Quaternary igneous centres (small curated list, abbreviated)
    volcanoes = pd.DataFrame([
        # Cascades
        ("Lassen", 40.4877, -121.5054), ("Shasta", 41.4090, -122.1949),
        ("Hood", 45.3735, -121.6959),  ("Crater Lake", 42.9446, -122.1090),
        ("Newberry", 43.6940, -121.2350), ("Glacier Peak", 48.1119, -121.1138),
        ("St Helens", 46.1992, -122.1882), ("Adams", 46.2024, -121.4909),
        # Long Valley / Coso / Inyo
        ("Long Valley", 37.7000, -118.8870), ("Coso", 36.0500, -117.8000),
        ("Mono Inyo", 37.8800, -119.0000),
        # Yellowstone / SRP volcanism
        ("Yellowstone", 44.4280, -110.5885), ("Craters of the Moon", 43.4170, -113.5170),
        # Imperial
        ("Salton Buttes", 33.1750, -115.6000),
        # New Mexico margin
        ("Valles caldera", 35.9000, -106.5300),
        # San Francisco volcanic field
        ("San Francisco volc field", 35.3500, -111.6800),
    ], columns=["name", "lat", "lon"])
    d_volc = nearest_distance_km(
        volcanoes[["lon", "lat"]].values, disc[["lon", "lat"]].values
    )
    disc["dist_to_nearest_volc_km"] = d_volc

    # Structural setting label
    disc["structural_setting"] = [
        structural_label(la, lo) for la, lo in zip(disc.lat, disc.lon)
    ]
    # System type from inferred T
    disc["system_type"] = [
        temperature_class(T) for T in disc["T_res_central"]
    ]

    # Order by P50 MWe
    cols = [
        "discovery_id", "lat", "lon", "province", "structural_setting",
        "system_type", "T_res_central", "area_km2", "n_cells",
        "p_mean", "p_std",
        "dist_to_max_km", "dist_to_nearest_qfault_km", "dist_to_nearest_volc_km",
        "mwe_p10", "mwe_p50", "mwe_p90",
    ]
    cols = [c for c in cols if c in disc.columns]
    out = disc[cols].sort_values("mwe_p50", ascending=False).reset_index(drop=True)

    out.to_csv(results / "discovery_geological_context.csv", index=False)
    print(f"wrote outputs/results/discovery_geological_context.csv")

    # Markdown table for the manuscript (top 10)
    top10 = out.head(10).copy()
    md = ["| Rank | Cluster | Lat | Lon | Province | Structural setting | T (deg C) | Type | Nearest field (km) | Nearest fault (km) | P50 MWe |",
          "|---:|---:|---:|---:|---|---|---:|---|---:|---:|---:|"]
    for i, r in top10.iterrows():
        md.append(f"| {i+1} | {int(r.discovery_id)} | {r.lat:.2f} | {r.lon:.2f} | "
                  f"{r.province} | {r.structural_setting} | "
                  f"{r.T_res_central:.0f} | {r.system_type.split(' (')[0]} | "
                  f"{r.dist_to_max_km:.1f} | {r.dist_to_nearest_qfault_km:.1f} | "
                  f"{r.mwe_p50:.0f} |")
    with open(results / "discovery_geological_context.md", "w") as f:
        f.write("\n".join(md))
    print(f"wrote outputs/results/discovery_geological_context.md")

    # Summary stats
    near_fault = float((out["dist_to_nearest_qfault_km"] <= 5.0).mean()) * 100
    near_volc = float((out["dist_to_nearest_volc_km"] <= 50.0).mean()) * 100
    print(f"\n  {near_fault:.0f}% of discoveries sit within 5 km of a mapped Q-fault")
    print(f"  {near_volc:.0f}% of discoveries sit within 50 km of a Quaternary volcanic centre")


if __name__ == "__main__":
    main()
