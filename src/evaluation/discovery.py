"""Continental geothermal discovery — train on all data, run MC-Dropout inference
across the full western-US grid, cluster high-confidence positives, and check
geological plausibility.

Outputs:
    outputs/results/prospectivity_mean.npy      [N] float32
    outputs/results/prospectivity_std.npy       [N] float32
    outputs/maps/prospectivity.tif              GeoTIFF for QGIS / ArcGIS
    outputs/results/table3_discoveries.csv      discovered cluster table
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

from src.data.dataset import GeoProspectDataset, make_loader
from src.data.grid import GridSpec, load_or_build_grid
from src.models.geoprospectnet import GeoProspectNet
from src.training.train import train_one_split
from src.training.utils import device_auto, load_config

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def _build_full_z_lookup(model, loader, device, fused_dim, n_total):
    """Single deterministic pass over the full grid to populate z_lookup
    used by spatial smoothing during the MC-Dropout passes."""
    model.eval()
    z_lookup = torch.zeros(n_total, fused_dim, device=device)
    for batch in loader:
        cell_ids = batch["cell_id"].to(device)
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        out = model(batch_dev)
        z_lookup[cell_ids] = out["z_fused"]
    return z_lookup


@torch.no_grad()
def mc_dropout_inference(model: torch.nn.Module, loader, device: str,
                         n_passes: int = 20,
                         neighbor_indices: np.ndarray | None = None,
                         z_lookup: torch.Tensor | None = None,
                         seed: int = 42):
    """Return mean and std of sigmoid(logits) across `n_passes` forward passes
    with dropout enabled. When ``neighbor_indices`` and ``z_lookup`` are
    supplied, spatial smoothing is applied at inference using the precomputed
    deterministic z_lookup.

    A per-pass torch RNG state is set from ``seed`` so the entire MC-Dropout
    sweep is bit-reproducible — successive runs produce identical mean and std
    arrays, which in turn produces identical discovery clusters and SHA-256
    pre-registration manifests.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.modules.dropout._DropoutNd):
            m.train()
    all_runs = []
    for pass_idx in range(n_passes):
        # Pin the RNG state for this pass so dropout masks are reproducible
        torch.manual_seed(seed + pass_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + pass_idx)
        np.random.seed(seed + pass_idx)
        scores = []
        for batch in loader:
            cell_ids = batch["cell_id"]
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            neighbor_z = None
            if neighbor_indices is not None and z_lookup is not None:
                idx = neighbor_indices[cell_ids.numpy()]
                neighbor_z = z_lookup[torch.as_tensor(idx, device=device,
                                                      dtype=torch.long)]
            out = model(batch_dev, neighbor_embeddings=neighbor_z)
            scores.append(torch.sigmoid(out["logits"]).cpu().numpy())
        all_runs.append(np.concatenate(scores))
    arr = np.stack(all_runs, axis=0)
    return arr.mean(0), arr.std(0)


def _province_bin(lat: float, lon: float) -> str:
    # Mirror of the function in build_labels.py (kept local to avoid import cycles).
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


def _plausibility(lat: float, lon: float, faults_xy, volcanoes_xy,
                  cos_lat0: float) -> str:
    px = lon * 111.32 * cos_lat0
    py = lat * 111.32
    prov = _province_bin(lat, lon)
    if prov in ("Basin_and_Range", "Cascades", "Snake_River_Plain", "Salton_Trough"):
        return "plausible"
    if faults_xy is not None and len(faults_xy):
        tree = cKDTree(faults_xy)
        d, _ = tree.query([[px, py]], k=1)
        if d[0] < 50.0:
            return "plausible"
    if volcanoes_xy is not None and len(volcanoes_xy):
        tree = cKDTree(volcanoes_xy)
        d, _ = tree.query([[px, py]], k=1)
        if d[0] < 100.0:
            return "plausible"
    return "implausible"


def _write_geotiff(out_path: Path, grid_df: pd.DataFrame, values: np.ndarray,
                   spec: GridSpec):
    """Write a single-band GeoTIFF on the regular lat/lon grid."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        print("[warn] rasterio not available; skipping GeoTIFF export")
        return
    n_rows = int(grid_df["row"].max()) + 1
    n_cols = int(grid_df["col"].max()) + 1
    img = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    img[grid_df["row"].values, grid_df["col"].values] = values
    # Origin = upper-left corner; row 0 is the SOUTH-most band in our grid (we
    # built with ascending lat), so flip vertically for north-up GeoTIFF.
    img = img[::-1, :]
    transform = from_origin(
        west=spec.lon_min, north=spec.lat_max,
        xsize=spec.lon_step_deg, ysize=spec.lat_step_deg,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=n_rows, width=n_cols,
        count=1, dtype="float32", crs="EPSG:4326", transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(img, 1)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--uncertainty", type=float, default=None)
    parser.add_argument("--min_samples", type=int, default=None)
    parser.add_argument("--retrain", action="store_true",
                        help="Re-train on full labeled data even if checkpoint exists")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a trained checkpoint. If omitted, uses "
                             "outputs/checkpoints/discovery_full.pt (training "
                             "it from scratch if missing).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    disc_cfg = cfg["discovery"]
    thr = args.threshold if args.threshold is not None else disc_cfg["prob_threshold"]
    unc = args.uncertainty if args.uncertainty is not None else disc_cfg["uncertainty_threshold"]
    min_samp = args.min_samples if args.min_samples is not None else disc_cfg["dbscan_min_samples"]

    processed = ROOT / cfg["paths"]["processed_dir"]
    grid_df = pd.read_csv(processed / "grid_coordinates.csv")
    sr = cfg["study_region"]
    spec = GridSpec(sr["lat_min"], sr["lat_max"], sr["lon_min"], sr["lon_max"],
                    sr["grid_resolution_km"])

    labels = np.load(processed / "labels.npy")
    train_mask = np.load(processed / "train_mask.npy")
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(train_mask & (labels == 0))[:min(2000, train_mask.sum() // 10)]
    train_idx = np.setdiff1d(train_idx, val_idx)

    # ---- Train (or load) ----
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--checkpoint {ckpt_path} not found")
    else:
        ckpt_path = ROOT / cfg["paths"]["checkpoints_dir"] / "discovery_full.pt"
        if args.retrain or not ckpt_path.exists():
            print("Training discovery model on full labeled data ...")
            r, ckpt_path = train_one_split(cfg, train_idx, val_idx, seed=42,
                                           device=device, log_prefix="discovery_full_")
    print(f"Using checkpoint: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GeoProspectNet(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # ---- MC-Dropout inference over the FULL grid ----
    print("Running MC-Dropout over the full grid ...")
    full_idx = np.arange(len(grid_df))
    full_ds = GeoProspectDataset(processed, indices=full_idx,
                                 use_thermal=cfg["modalities"]["use_thermal"])
    full_loader = make_loader(full_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=cfg["training"]["num_workers"])

    # Spatial smoothing is enabled at inference: one deterministic pass to
    # build a z_lookup table over all 247k cells, then MC-Dropout passes
    # gather neighbour embeddings from that table.
    neighbor_indices = None
    z_lookup = None
    nbr_path = processed / "neighbor_indices.npy"
    if cfg["model"]["spatial_alpha"] > 0 and nbr_path.exists():
        print("  building deterministic z_lookup for spatial smoothing ...")
        neighbor_indices = np.load(nbr_path)
        z_lookup = _build_full_z_lookup(
            model, full_loader, device,
            fused_dim=cfg["model"]["fused_dim"],
            n_total=len(grid_df),
        )

    mean_p, std_p = mc_dropout_inference(
        model, full_loader, device,
        n_passes=disc_cfg["mc_dropout_passes"],
        neighbor_indices=neighbor_indices, z_lookup=z_lookup,
    )

    np.save(ROOT / "outputs/results/prospectivity_mean.npy", mean_p)
    np.save(ROOT / "outputs/results/prospectivity_std.npy", std_p)
    _write_geotiff(ROOT / "outputs/maps/prospectivity.tif", grid_df, mean_p, spec)
    _write_geotiff(ROOT / "outputs/maps/uncertainty.tif", grid_df, std_p, spec)

    # ---- Discovery filtering ----
    print(f"Filtering: p>{thr}  std<{unc}")
    candidate = (mean_p > thr) & (std_p < unc)

    # Exclude grid cells within `exclusion_radius_km` of known fields
    meta = ROOT / cfg["paths"]["metadata_dir"]
    fields_path = meta / "known_fields_details.csv"
    if not fields_path.exists():
        print(f"[warn] {fields_path} missing — skipping known-field exclusion")
        fields = pd.DataFrame(columns=["lat", "lon"])
    else:
        fields = pd.read_csv(fields_path)
    lat0 = float(grid_df["lat"].mean())
    cos_lat0 = float(np.cos(np.radians(lat0)))
    g_xy = np.column_stack([
        grid_df["lon"].values * 111.32 * cos_lat0,
        grid_df["lat"].values * 111.32,
    ])
    if len(fields):
        f_xy = np.column_stack([
            fields["lon"].values * 111.32 * cos_lat0,
            fields["lat"].values * 111.32,
        ])
        tree = cKDTree(f_xy)
        d, _ = tree.query(g_xy, k=1)
        far_enough = d > disc_cfg["exclusion_radius_km"]
        candidate &= far_enough

    print(f"Candidate cells after filtering: {candidate.sum()}")

    # ---- DBSCAN clustering on candidate cells ----
    cand_xy = g_xy[candidate]
    if len(cand_xy) >= min_samp:
        db = DBSCAN(eps=disc_cfg["dbscan_eps_km"], min_samples=min_samp).fit(cand_xy)
        cluster_id = db.labels_
    else:
        cluster_id = np.full(len(cand_xy), -1)

    cand_df = grid_df.iloc[candidate].reset_index(drop=True).copy()
    cand_df["p_mean"] = mean_p[candidate]
    cand_df["p_std"] = std_p[candidate]
    cand_df["cluster"] = cluster_id

    # ---- Aggregate clusters into discoveries ----
    fault_xy = None
    fault_csv = meta / "fault_points.csv"  # optional precomputed
    if fault_csv.exists():
        fp = pd.read_csv(fault_csv)
        fault_xy = np.column_stack([fp.lon * 111.32 * cos_lat0, fp.lat * 111.32])
    volcanoes_path = ROOT / "data/raw/geology/holocene_volcanoes.csv"
    volc_xy = None
    if volcanoes_path.exists():
        v = pd.read_csv(volcanoes_path)
        v.columns = [c.lower().strip() for c in v.columns]
        lat_c = next((c for c in v.columns if "lat" in c), None)
        lon_c = next((c for c in v.columns if "lon" in c), None)
        if lat_c and lon_c:
            v = v[[lat_c, lon_c]].dropna()
            volc_xy = np.column_stack([
                pd.to_numeric(v[lon_c], errors="coerce") * 111.32 * cos_lat0,
                pd.to_numeric(v[lat_c], errors="coerce") * 111.32,
            ])

    discoveries = []
    for cid in sorted(set(int(c) for c in cluster_id if c >= 0)):
        cluster = cand_df[cand_df["cluster"] == cid]
        center_lat = float(cluster["lat"].mean())
        center_lon = float(cluster["lon"].mean())
        area_km2 = float(len(cluster)) * (spec.resolution_km ** 2)
        plaus = _plausibility(center_lat, center_lon, fault_xy, volc_xy, cos_lat0)
        discoveries.append({
            "discovery_id": cid,
            "lat": center_lat,
            "lon": center_lon,
            "area_km2": area_km2,
            "p_mean": float(cluster["p_mean"].mean()),
            "p_std": float(cluster["p_std"].mean()),
            "n_cells": int(len(cluster)),
            "province": _province_bin(center_lat, center_lon),
            "plausibility": plaus,
        })
    disc_df = pd.DataFrame(discoveries).sort_values("p_mean", ascending=False).reset_index(drop=True)
    out_csv = ROOT / "outputs/results/table3_discoveries.csv"
    disc_df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(disc_df)} discoveries)")
    if len(disc_df):
        share = (disc_df["plausibility"] == "plausible").mean()
        print(f"Plausible share: {share:.1%}")


if __name__ == "__main__":
    main()
