"""Out-of-distribution test on the eastern United States — in-memory scoring,
no large disk writes.

Builds a 4 km grid east of the Rockies, ingests gravity / mag / heat flow /
elevation from public rasters, and streams each cell through the cpu_max
model. Looks up scores at documented eastern hydrothermal / warm-spring
systems (Hot Springs AR, Warm Springs GA, Berkeley Springs WV, etc.).

Geophysics is built as cell-centre point values broadcast to a uniform
32x32 patch *inside the inference loop* (never materialised on disk).
Geochemistry and geology modalities are masked False because eastern
coverage was not part of the training pipeline.

Outputs:
    outputs/results/ood_eastern_us_scores.npy    [N] float32
    outputs/results/ood_eastern_us_grid.csv      grid + score per cell
    outputs/results/ood_eastern_us_validation.csv per-known-site lookup
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]

KNOWN_EASTERN_SITES = pd.DataFrame([
    dict(name="Hot Springs, AR",        lat=34.5117, lon=-93.0531,  T_C=62, type="thermal"),
    dict(name="Warm Springs, GA",       lat=32.8893, lon=-84.6810,  T_C=33, type="warm"),
    dict(name="Berkeley Springs, WV",   lat=39.6262, lon=-78.2278,  T_C=22, type="warm"),
    dict(name="Lebanon Springs, NY",    lat=42.4673, lon=-73.3937,  T_C=22, type="warm"),
    dict(name="Saratoga Springs, NY",   lat=43.0831, lon=-73.7846,  T_C=10, type="cool_spring"),
    dict(name="Hot Springs, VA",        lat=38.0026, lon=-79.8333,  T_C=42, type="thermal"),
    dict(name="Bedford Springs, PA",    lat=40.0049, lon=-78.5023,  T_C=18, type="warm"),
    dict(name="Mt Princeton Hot Spr CO", lat=38.7297, lon=-106.1614, T_C=55, type="thermal"),  # control: should still score high
])


def build_grid(lon_min: float, lon_max: float, lat_min: float, lat_max: float,
                resolution_km: float = 4.0) -> pd.DataFrame:
    mid_lat = 0.5 * (lat_min + lat_max)
    dlat = resolution_km / 111.32
    dlon = resolution_km / (111.32 * max(0.2, np.cos(np.radians(mid_lat))))
    lats = np.arange(lat_min, lat_max + 1e-6, dlat)
    lons = np.arange(lon_min, lon_max + 1e-6, dlon)
    LL, LN = np.meshgrid(lats, lons, indexing="ij")
    df = pd.DataFrame({"lat": LL.ravel(), "lon": LN.ravel()})
    df["cell_id"] = np.arange(len(df))
    return df


def sample_raster(tif_path: Path, lats, lons, default=np.nan,
                   bounds_override=None):
    if not tif_path.exists():
        return np.full(len(lats), default, dtype=np.float32)
    with rasterio.open(tif_path) as src:
        if bounds_override is not None:
            bw, bs, be, bn = bounds_override
        else:
            bw, bs, be, bn = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
        arr = src.read(1).astype(np.float32)
        nx, ny = src.width, src.height
        out = np.full(len(lats), default, dtype=np.float32)
        for i, (la, lo) in enumerate(zip(lats, lons)):
            if lo < bw or lo > be or la < bs or la > bn:
                continue
            col = int((lo - bw) / (be - bw) * nx)
            row = int((bn - la) / (bn - bs) * ny)
            col = max(0, min(nx - 1, col))
            row = max(0, min(ny - 1, row))
            v = arr[row, col]
            if np.isfinite(v):
                out[i] = float(v)
    return out


def sample_heat_flow(lats, lons):
    hf = pd.read_csv(ROOT / "data/raw/geophysics/smu_heatflow/heat_flow_combined.csv",
                      usecols=["LatDegreeWGS84", "LongDegreeWGS84", "SiteHeatFlow"])
    hf = hf.dropna(subset=["LatDegreeWGS84", "LongDegreeWGS84"])
    hf["SiteHeatFlow"] = pd.to_numeric(hf["SiteHeatFlow"], errors="coerce")
    hf = hf[(hf["SiteHeatFlow"] > 0) & (hf["SiteHeatFlow"] < 500)]
    lat0 = float(np.mean(lats))
    cos_lat0 = float(np.cos(np.radians(lat0)))
    hf_xy = np.column_stack([hf.LongDegreeWGS84 * 111.32 * cos_lat0,
                              hf.LatDegreeWGS84 * 111.32])
    tree = cKDTree(hf_xy)
    grid_xy = np.column_stack([np.asarray(lons) * 111.32 * cos_lat0,
                                np.asarray(lats) * 111.32])
    d, idx = tree.query(grid_xy, k=5)
    weights = 1.0 / (d + 1.0)
    weighted = (hf.SiteHeatFlow.values[idx] * weights).sum(axis=1) / weights.sum(axis=1)
    far = d[:, 0] > 200
    weighted[far] = np.nan
    return weighted.astype(np.float32)


def build_cell_features(grid: pd.DataFrame):
    bouguer = sample_raster(ROOT / "data/raw/geophysics/bouguer_gravity.tif",
                            grid.lat.values, grid.lon.values, default=0.0,
                            bounds_override=(-124.7, 25.1, -65.7, 52.0))
    isostatic = sample_raster(ROOT / "data/raw/geophysics/isostatic_gravity.tif",
                              grid.lat.values, grid.lon.values, default=0.0,
                              bounds_override=(-124.7, 25.1, -65.7, 52.0))
    elev = sample_raster(ROOT / "data/raw/geology/srtm_elevation.tif",
                          grid.lat.values, grid.lon.values, default=0.0)
    hf = sample_heat_flow(grid.lat.values, grid.lon.values)
    hf_med = float(np.nanmedian(hf))
    hf = np.nan_to_num(hf, nan=hf_med)

    def z(arr):
        m = np.nanmean(arr); s = np.nanstd(arr) + 1e-6
        return ((arr - m) / s).astype(np.float32)

    c0 = z(bouguer); c1 = z(isostatic); c2 = np.zeros_like(c0)
    c3 = z(hf); c4 = z(elev); c5 = np.zeros_like(c0)
    return np.stack([c0, c1, c2, c3, c4, c5], axis=1)  # [N, 6]


def stream_score(cell_features: np.ndarray, batch_size: int = 256):
    """Score in batches: broadcast cell_features to (B, 6, 32, 32) on the fly."""
    import sys; sys.path.insert(0, str(ROOT))
    from src.models.geoprospectnet import GeoProspectNet
    cfg = yaml.safe_load(open(ROOT / "configs/cpu_max.yaml"))
    ck = torch.load(ROOT / "outputs/checkpoints/random_cpu_max_seed42.pt",
                     map_location="cpu", weights_only=False)
    model = GeoProspectNet(cfg).to("cpu"); model.load_state_dict(ck["state_dict"]); model.eval()

    n = len(cell_features)
    scores = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            ce = torch.from_numpy(cell_features[i:j])           # [B, 6]
            patches = ce[:, :, None, None].expand(-1, -1, 32, 32).contiguous()
            geo_mask = torch.ones(j - i, dtype=torch.bool)
            geochem = torch.zeros(j - i, 9)
            chem_mask = torch.zeros(j - i, dtype=torch.bool)
            geology = torch.zeros(j - i, 11)
            struct_mask = torch.zeros(j - i, dtype=torch.bool)
            cell_id = torch.arange(i, j, dtype=torch.long)
            batch = {
                "geophysics": patches,
                "geo_mask": geo_mask,
                "geochemistry": geochem,
                "chem_mask": chem_mask,
                "geology": geology,
                "struct_mask": struct_mask,
                "cell_id": cell_id,
                "label": torch.zeros(j - i),
            }
            out = model(batch)
            scores[i:j] = torch.sigmoid(out["logits"]).numpy()
            if i % (batch_size * 50) == 0:
                print(f"  scored {j}/{n}", flush=True)
    return scores


def main():
    grid = build_grid(lon_min=-103.0, lon_max=-67.0, lat_min=25.0, lat_max=49.0,
                      resolution_km=4.0)
    print(f"Eastern grid: {len(grid):,} cells", flush=True)

    print("ingesting geophysics (gravity, magnetic, HF, elevation) ...", flush=True)
    cell_feat = build_cell_features(grid)
    print(f"  features shape: {cell_feat.shape}", flush=True)

    print("streaming through cpu_max model ...", flush=True)
    scores = stream_score(cell_feat, batch_size=256)
    np.save(ROOT / "outputs/results/ood_eastern_us_scores.npy", scores)
    grid["score"] = scores

    # Save grid with score (compact)
    grid[["cell_id", "lat", "lon", "score"]].to_csv(
        ROOT / "outputs/results/ood_eastern_us_grid.csv", index=False)
    print(f"eastern median={np.median(scores):.4f}  mean={scores.mean():.4f}  "
          f"frac>0.5={(scores>0.5).mean():.1%}", flush=True)

    # Validate at known sites
    lat0 = float(grid.lat.mean()); cos_lat0 = float(np.cos(np.radians(lat0)))
    gxy = np.column_stack([grid.lon * 111.32 * cos_lat0, grid.lat * 111.32])
    tree = cKDTree(gxy)
    sort = np.argsort(scores)
    rank = np.empty_like(sort, dtype=np.int64); rank[sort] = np.arange(len(scores))
    pct = 100.0 * rank / (len(scores) - 1)

    rows = []
    for _, s in KNOWN_EASTERN_SITES.iterrows():
        d, i = tree.query([s.lon * 111.32 * cos_lat0, s.lat * 111.32], k=1)
        rows.append({
            "name": s["name"], "lat": s.lat, "lon": s.lon,
            "T_C": s.T_C, "type": s.type,
            "nearest_dist_km": float(d),
            "score": float(scores[i]),
            "percentile": float(pct[i]),
        })
    df = pd.DataFrame(rows).sort_values("percentile", ascending=False)
    out = ROOT / "outputs/results/ood_eastern_us_validation.csv"
    df.to_csv(out, index=False)
    print("\n=== OOD VALIDATION at known eastern systems ===")
    print(df.to_string(index=False, float_format="%.2f"))
    print()
    east = df[df.lon > -100]
    if len(east):
        print(f"Eastern sites only ({len(east)} of {len(df)}):")
        print(f"  Mean percentile: {east.percentile.mean():.1f}")
        print(f"  Median:          {east.percentile.median():.1f}")
        print(f"  Fraction in top 25%: {(east.percentile >= 75).mean():.1%}")
        print(f"  Fraction in top 10%: {(east.percentile >= 90).mean():.1%}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
