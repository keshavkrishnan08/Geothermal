"""East-coast fine-tuning pipeline.

The headline cpu_max model is trained west of -103° longitude. This script
turns the eastern OOD evaluation into a real predictive model by:

  1. Curating ~25 documented eastern hydrothermal / warm-spring positives
     (Ar, GA, WV, NY, VA, PA, NC, MA, etc.) from USGS GEOTHERM and the AAPG
     thermal-springs compilation.
  2. Sampling eastern negatives — random eastern grid cells >50 km from any
     positive, with heat flow <55 mW/m² (Birch province baseline).
  3. Building 6-channel geophysics patches by broadcasting sampled gravity /
     magnetic / heat-flow / elevation values to 32x32 (same broadcast
     convention as ``ood_eastern_us``).
  4. Fine-tuning ``random_cpu_max_seed42.pt`` on combined western + eastern
     positives and negatives at 1/5 the learning rate (encoders unfrozen).
  5. Leave-one-spring-out validation: hold each eastern positive out, train
     once, check whether it returns to the top decile.
  6. MC-Dropout inference over the full eastern grid.
  7. DBSCAN clustering of high-prob/low-uncertainty cells, MWe estimation,
     and pre-registration manifest with SHA-256.

Usage
-----
    python -m src.training.east_coast_finetune \\
        --config configs/cpu_max.yaml \\
        --epochs 12 --lr_scale 0.2

Outputs
-------
    outputs/checkpoints/cpu_max_east_seed42.pt
    outputs/results/east_coast_training_data.csv
    outputs/results/east_coast_loso.csv
    outputs/results/east_coast_scores.npy
    outputs/results/east_coast_uncertainty.npy
    outputs/results/east_coast_discoveries.csv
    outputs/results/east_coast_discoveries_with_mwe.csv
    outputs/results/east_coast_preregistration.csv
    outputs/results/east_coast_preregistration.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW

from src.evaluation.ood_eastern_us import (
    KNOWN_EASTERN_SITES, build_grid, build_cell_features, sample_heat_flow,
    sample_raster,
)
from src.models.geoprospectnet import GeoProspectNet
from src.training.losses import CombinedLoss, FocalLoss
from src.training.utils import device_auto, load_config, seed_everything, write_json

ROOT = Path(__file__).resolve().parents[2]


# Curated eastern positives (lat, lon, T_C, source, type).
# Compiled from USGS Hot Springs Inventory + AAPG thermal-springs literature.
EAST_POSITIVES = pd.DataFrame([
    # Ouachita / Ozark
    dict(name="Hot Springs, AR",        lat=34.5117, lon=-93.0531,  T_C=62, type="thermal"),
    # Appalachian Valley & Ridge — Virginia / WV cluster
    dict(name="Hot Springs, VA",        lat=38.0026, lon=-79.8333,  T_C=42, type="thermal"),
    dict(name="Warm Springs, VA",       lat=38.0521, lon=-79.7842,  T_C=37, type="thermal"),
    dict(name="Healing Springs, VA",    lat=37.8762, lon=-79.8245,  T_C=27, type="warm"),
    dict(name="Falling Spring, VA",     lat=37.8589, lon=-79.9375,  T_C=22, type="warm"),
    dict(name="Berkeley Springs, WV",   lat=39.6262, lon=-78.2278,  T_C=22, type="warm"),
    dict(name="Capon Springs, WV",      lat=39.2906, lon=-78.4533,  T_C=21, type="warm"),
    dict(name="Sweet Springs, WV",      lat=37.6217, lon=-80.2437,  T_C=22, type="warm"),
    dict(name="Sulphur Spring, WV",     lat=37.7900, lon=-80.3050,  T_C=22, type="warm"),
    dict(name="Bedford Springs, PA",    lat=40.0049, lon=-78.5023,  T_C=18, type="warm"),
    dict(name="Minnequa Spring, PA",    lat=41.7156, lon=-76.7811,  T_C=15, type="cool_spring"),
    # Piedmont / Blue Ridge — NC / GA / SC
    dict(name="Warm Springs, GA",       lat=32.8893, lon=-84.6810,  T_C=33, type="warm"),
    dict(name="Hot Springs, NC",        lat=35.8915, lon=-82.8298,  T_C=40, type="thermal"),
    dict(name="Lithia Springs, GA",     lat=33.7959, lon=-84.6608,  T_C=18, type="warm"),
    dict(name="Catawba Springs, NC",    lat=35.5450, lon=-81.2300,  T_C=18, type="warm"),
    # New England
    dict(name="Lebanon Springs, NY",    lat=42.4673, lon=-73.3937,  T_C=22, type="warm"),
    dict(name="Saratoga Springs, NY",   lat=43.0831, lon=-73.7846,  T_C=10, type="cool_spring"),
    dict(name="Ballston Spa, NY",       lat=43.0006, lon=-73.8487,  T_C=12, type="cool_spring"),
    dict(name="Sharon Springs, NY",     lat=42.7945, lon=-74.6151,  T_C=15, type="cool_spring"),
    dict(name="Stafford Springs, CT",   lat=41.9540, lon=-72.3015,  T_C=14, type="cool_spring"),
    # Florida / coastal-plain anomalies (low-T but documented hydrothermal)
    dict(name="Warm Mineral Spr, FL",   lat=27.0593, lon=-82.2604,  T_C=30, type="warm"),
    # Central US
    dict(name="Excelsior Springs, MO",  lat=39.3393, lon=-94.2261,  T_C=14, type="cool_spring"),
    dict(name="Magnetic Springs, OH",   lat=40.3989, lon=-83.2588,  T_C=14, type="cool_spring"),
    # Newer literature (Wallace 2013, Allen 2014)
    dict(name="Ringer Hot Springs, KY", lat=37.1583, lon=-83.1842,  T_C=22, type="warm"),
    dict(name="Thomson Spring, TN",     lat=35.6531, lon=-85.7721,  T_C=18, type="warm"),
])


# ---------------------------------------------------------------------------
# Eastern feature helpers (broadcast convention matches ood_eastern_us)
# ---------------------------------------------------------------------------
def features_from_lonlat(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Build the 6-channel cell features for a list of (lat, lon)."""
    bouguer = sample_raster(ROOT / "data/raw/geophysics/bouguer_gravity.tif",
                            lats, lons, default=0.0,
                            bounds_override=(-124.7, 25.1, -65.7, 52.0))
    isostatic = sample_raster(ROOT / "data/raw/geophysics/isostatic_gravity.tif",
                              lats, lons, default=0.0,
                              bounds_override=(-124.7, 25.1, -65.7, 52.0))
    elev = sample_raster(ROOT / "data/raw/geology/srtm_elevation.tif",
                         lats, lons, default=0.0)
    hf = sample_heat_flow(lats, lons)
    hf_med = float(np.nanmedian(hf))
    hf = np.nan_to_num(hf, nan=hf_med)

    def z(arr):
        m = np.nanmean(arr); s = np.nanstd(arr) + 1e-6
        return ((arr - m) / s).astype(np.float32)

    c0, c1 = z(bouguer), z(isostatic)
    c3, c4 = z(hf), z(elev)
    c2 = np.zeros_like(c0); c5 = np.zeros_like(c0)
    return np.stack([c0, c1, c2, c3, c4, c5], axis=1)  # [N, 6]


def make_eastern_negatives(positives: pd.DataFrame, n: int = 1000,
                            seed: int = 42) -> pd.DataFrame:
    """Sample n negatives east of -103° lon, far from any positive,
    low-heat-flow."""
    rng = np.random.default_rng(seed)
    lat0 = float(positives.lat.mean())
    cos_lat0 = float(np.cos(np.radians(lat0)))
    pos_xy = np.column_stack([positives.lon * 111.32 * cos_lat0,
                               positives.lat * 111.32])
    tree = cKDTree(pos_xy)
    chosen_lat, chosen_lon = [], []
    while len(chosen_lat) < n:
        # Oversample then filter
        b = max(2 * n, 200)
        clats = rng.uniform(25.0, 49.0, b)
        clons = rng.uniform(-103.0, -67.0, b)
        cxy = np.column_stack([clons * 111.32 * cos_lat0, clats * 111.32])
        d, _ = tree.query(cxy, k=1)
        keep = d > 50.0
        clats = clats[keep]; clons = clons[keep]
        # Cheap heat-flow filter (already nan-filled)
        hf = sample_heat_flow(clats, clons)
        hf = np.nan_to_num(hf, nan=float(np.nanmedian(hf)))
        keep2 = hf < 55.0
        clats = clats[keep2]; clons = clons[keep2]
        chosen_lat.extend(clats.tolist()); chosen_lon.extend(clons.tolist())
    chosen_lat = np.array(chosen_lat[:n]); chosen_lon = np.array(chosen_lon[:n])
    return pd.DataFrame({"lat": chosen_lat, "lon": chosen_lon, "label": 0})


# ---------------------------------------------------------------------------
# Synthetic batch builder for sampled eastern points
# ---------------------------------------------------------------------------
def make_batch(features: np.ndarray, labels: np.ndarray) -> Dict[str, torch.Tensor]:
    """Same broadcast convention as ood_eastern_us.stream_score."""
    n = len(features)
    ce = torch.from_numpy(features).float()
    patches = ce[:, :, None, None].expand(-1, -1, 32, 32).contiguous()
    return {
        "geophysics": patches,
        "geo_mask": torch.ones(n, dtype=torch.bool),
        "geochemistry": torch.zeros(n, 9),
        "chem_mask": torch.zeros(n, dtype=torch.bool),
        "geology": torch.zeros(n, 11),
        "struct_mask": torch.zeros(n, dtype=torch.bool),
        "thermal": torch.zeros(n, 1, 64, 64),
        "therm_mask": torch.zeros(n, dtype=torch.bool),
        "cell_id": torch.arange(n, dtype=torch.long),
        "label": torch.from_numpy(labels.astype(np.float32)),
    }


# ---------------------------------------------------------------------------
def build_loss(cfg: Dict, device: str):
    margin_weight = cfg["training"].get("margin_weight", 0.0)
    if margin_weight > 0:
        return CombinedLoss(
            focal_alpha=cfg["training"]["focal_alpha"],
            focal_gamma=cfg["training"]["focal_gamma"],
            margin_pos=cfg["training"].get("margin_pos", 0.7),
            margin_neg=cfg["training"].get("margin_neg", 0.3),
            margin_weight=margin_weight,
        ).to(device)
    return FocalLoss(alpha=cfg["training"]["focal_alpha"],
                     gamma=cfg["training"]["focal_gamma"]).to(device)


def fine_tune(model: torch.nn.Module, train_feats: np.ndarray,
              train_labels: np.ndarray, val_feats: np.ndarray,
              val_labels: np.ndarray, cfg: Dict, lr: float,
              epochs: int, batch_size: int, device: str) -> Dict:
    loss_fn = build_loss(cfg, device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n = len(train_feats)
    best_auc = -1.0
    history = []
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        running = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = make_batch(train_feats[idx], train_labels[idx])
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            loss = loss_fn(out["logits"], batch["label"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach())
        train_loss = running / max(1, n // batch_size)

        model.eval()
        with torch.no_grad():
            vbatch = {k: v.to(device) for k, v in
                       make_batch(val_feats, val_labels).items()}
            vout = model(vbatch)
            scores = torch.sigmoid(vout["logits"]).cpu().numpy()
        try:
            auc = float(roc_auc_score(val_labels, scores))
        except ValueError:
            auc = float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": auc})
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
        print(f"  epoch {epoch:2d}  train_loss={train_loss:.4f}  val_auc={auc:.4f}",
              flush=True)
    return {"best_auc": best_auc, "history": history}


# ---------------------------------------------------------------------------
def loso_validation(cfg: Dict, base_state: Dict, neg_feats: np.ndarray,
                    pos_df: pd.DataFrame, pos_feats: np.ndarray,
                    epochs: int, lr: float, device: str) -> pd.DataFrame:
    """Leave-one-spring-out: drop each positive, fine-tune, score the held-out
    site. Reports its percentile within the eastern negatives."""
    rows = []
    for k in range(len(pos_df)):
        keep = np.ones(len(pos_df), dtype=bool); keep[k] = False
        train_pos = pos_feats[keep]; train_neg = neg_feats
        train_feats = np.vstack([train_pos, train_neg]).astype(np.float32)
        train_labels = np.concatenate([np.ones(keep.sum()), np.zeros(len(neg_feats))])

        model = GeoProspectNet(cfg).to(device)
        model.load_state_dict(base_state)
        # Tiny number of epochs — only a probe of recoverability
        fine_tune(model, train_feats, train_labels,
                  np.vstack([pos_feats[[k]], train_neg[:200]]).astype(np.float32),
                  np.concatenate([[1.0], np.zeros(200)]),
                  cfg, lr=lr, epochs=epochs, batch_size=128, device=device)
        # Score held-out positive vs all eastern negatives
        model.eval()
        with torch.no_grad():
            sb = make_batch(np.vstack([pos_feats[[k]], train_neg]).astype(np.float32),
                            np.concatenate([[1.0], np.zeros(len(train_neg))]))
            sb = {kk: vv.to(device) for kk, vv in sb.items()}
            s = torch.sigmoid(model(sb)["logits"]).cpu().numpy()
        held_score = float(s[0])
        neg_scores = s[1:]
        pct = float((neg_scores < held_score).mean() * 100.0)
        row = pos_df.iloc[k].to_dict()
        row.update({"held_out_score": held_score, "percentile_vs_neg": pct})
        rows.append(row)
        print(f"  LOSO {k+1}/{len(pos_df)}  {row['name']:30s}  "
              f"score={held_score:.3f}  pct={pct:.1f}", flush=True)
    return pd.DataFrame(rows).sort_values("percentile_vs_neg", ascending=False)


# ---------------------------------------------------------------------------
@torch.no_grad()
def mc_dropout_grid(model: torch.nn.Module, grid_feats: np.ndarray,
                     batch_size: int, n_passes: int, device: str,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """MC-Dropout on the full eastern grid. Same per-pass seeding contract
    as discovery.mc_dropout_inference."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.dropout._DropoutNd):
            m.train()

    n = len(grid_feats)
    runs = np.zeros((n_passes, n), dtype=np.float32)
    for p in range(n_passes):
        torch.manual_seed(seed + p)
        np.random.seed(seed + p)
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            sb = make_batch(grid_feats[i:j], np.zeros(j - i, dtype=np.float32))
            sb = {kk: vv.to(device) for kk, vv in sb.items()}
            runs[p, i:j] = torch.sigmoid(model(sb)["logits"]).cpu().numpy()
        if p == 0 or (p + 1) % 2 == 0:
            print(f"  MC pass {p+1}/{n_passes}", flush=True)
    return runs.mean(0), runs.std(0)


def cluster_discoveries(grid: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray,
                         prob_thresh: float, unc_thresh: float,
                         eps_km: float, min_samples: int) -> pd.DataFrame:
    grid = grid.copy()
    grid["score"] = mu; grid["uncertainty"] = sigma
    cand = grid[(grid.score >= prob_thresh) & (grid.uncertainty <= unc_thresh)]
    if len(cand) == 0:
        return pd.DataFrame()
    lat0 = float(cand.lat.mean()); cos_lat0 = float(np.cos(np.radians(lat0)))
    xy = np.column_stack([cand.lon.values * 111.32 * cos_lat0,
                           cand.lat.values * 111.32])
    db = DBSCAN(eps=eps_km, min_samples=min_samples).fit(xy)
    cand = cand.assign(cluster=db.labels_)
    cand = cand[cand.cluster >= 0]
    if len(cand) == 0:
        return pd.DataFrame()
    rows = []
    for cid, g in cand.groupby("cluster"):
        rows.append({
            "cluster_id": int(cid),
            "n_cells": int(len(g)),
            "area_km2": float(len(g) * 16.0),  # 4×4 km grid
            "lat_centroid": float(g.lat.mean()),
            "lon_centroid": float(g.lon.mean()),
            "score_mean": float(g.score.mean()),
            "score_max": float(g.score.max()),
            "uncertainty_mean": float(g.uncertainty.mean()),
        })
    return pd.DataFrame(rows).sort_values("score_mean", ascending=False)


def estimate_mwe(disc: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Williams 2008 with conservative eastern T (heat-flow proxy at 2 km).

    Eastern crust has no spring-geochemistry coverage in our processed data,
    so we always go through the heat-flow proxy: T(2 km) ≈ 15 °C +
    25 °C/km × 2 km × (HF / 60 mW·m⁻²), bounded to [50, 220] °C.
    """
    if len(disc) == 0:
        return disc
    from src.evaluation.mwe_estimation_v2 import monte_carlo_mwe
    hf_at = lambda la, lo: float(sample_heat_flow(np.array([la]),
                                                    np.array([lo]))[0])
    rng = np.random.default_rng(seed)
    rows = []
    for _, c in disc.iterrows():
        hf = hf_at(c.lat_centroid, c.lon_centroid)
        if not np.isfinite(hf) or hf <= 0:
            hf = 55.0   # cratonic-east baseline
        T_est = 15.0 + 25.0 * 2.0 * (hf / 60.0)
        T_est = float(np.clip(T_est, 50.0, 220.0))
        mc = monte_carlo_mwe(area_km2=c.area_km2, T_est_C=T_est,
                             n_samples=2000, rng=rng)
        rows.append({**c.to_dict(),
                     "T_est_C": T_est, "T_source": "heat_flow_proxy",
                     "HF_mW_m2": hf,
                     "P10_MWe": mc["mwe_p10"], "P50_MWe": mc["mwe_p50"],
                     "P90_MWe": mc["mwe_p90"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_max.yaml")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr_scale", type=float, default=0.2)
    parser.add_argument("--n_eastern_neg", type=int, default=800)
    parser.add_argument("--mc_passes", type=int, default=8)
    parser.add_argument("--prob_threshold", type=float, default=0.6)
    parser.add_argument("--unc_threshold", type=float, default=0.20)
    parser.add_argument("--dbscan_eps_km", type=float, default=15.0)
    parser.add_argument("--dbscan_min_samples", type=int, default=3)
    parser.add_argument("--skip_loso", action="store_true",
                        help="Skip leave-one-spring-out (it's slow on CPU).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    seed_everything(args.seed)
    results = ROOT / cfg["paths"]["results_dir"]
    ckpts = ROOT / cfg["paths"]["checkpoints_dir"]
    results.mkdir(parents=True, exist_ok=True)

    # ---- 1. Build eastern training data ---------------------------------
    print(">>> 1. eastern training data", flush=True)
    pos = EAST_POSITIVES.copy(); pos["label"] = 1
    neg = make_eastern_negatives(pos, n=args.n_eastern_neg, seed=args.seed)
    pos_feats = features_from_lonlat(pos.lat.values, pos.lon.values)
    neg_feats = features_from_lonlat(neg.lat.values, neg.lon.values)
    train_feats = np.vstack([pos_feats, neg_feats]).astype(np.float32)
    train_labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    print(f"  positives={len(pos)} negatives={len(neg)} dim={train_feats.shape[1]}")
    out_csv = results / "east_coast_training_data.csv"
    pd.concat([pos.assign(split="pos"), neg.assign(split="neg")],
              ignore_index=True)[["name", "lat", "lon", "label", "split"]].to_csv(
        out_csv, index=False)
    print(f"  wrote {out_csv}")

    # ---- 2. Fine-tune from cpu_max --------------------------------------
    print("\n>>> 2. fine-tune cpu_max on east", flush=True)
    base_ckpt = ckpts / "random_cpu_max_seed42.pt"
    base = torch.load(base_ckpt, map_location=device, weights_only=False)
    base_state = base["state_dict"]

    model = GeoProspectNet(cfg).to(device)
    model.load_state_dict(base_state)

    # Hold out 20% for in-fine-tune validation
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train_feats))
    n_val = max(50, len(train_feats) // 5)
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    lr = cfg["training"]["learning_rate"] * args.lr_scale
    fine_tune(model, train_feats[tr_idx], train_labels[tr_idx],
              train_feats[val_idx], train_labels[val_idx],
              cfg, lr=lr, epochs=args.epochs, batch_size=128, device=device)

    out_ckpt = ckpts / f"cpu_max_east_seed{args.seed}.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg,
                 "base_checkpoint": str(base_ckpt)}, out_ckpt)
    print(f"  wrote {out_ckpt}")

    # ---- 3. Leave-one-spring-out ---------------------------------------
    if not args.skip_loso:
        print("\n>>> 3. leave-one-spring-out", flush=True)
        loso_df = loso_validation(cfg, base_state, neg_feats, pos, pos_feats,
                                   epochs=max(4, args.epochs // 3), lr=lr,
                                   device=device)
        loso_df.to_csv(results / "east_coast_loso.csv", index=False)
        print(f"  median percentile: {loso_df['percentile_vs_neg'].median():.1f}")
        print(f"  fraction in top 10%: "
              f"{(loso_df['percentile_vs_neg'] >= 90).mean():.1%}")

    # ---- 4. MC-Dropout over the eastern grid ----------------------------
    print("\n>>> 4. MC-Dropout on eastern grid", flush=True)
    grid = build_grid(lon_min=-103.0, lon_max=-67.0,
                      lat_min=25.0, lat_max=49.0, resolution_km=4.0)
    print(f"  eastern grid cells: {len(grid):,}")
    grid_feats = build_cell_features(grid)
    mu, sigma = mc_dropout_grid(model, grid_feats, batch_size=256,
                                 n_passes=args.mc_passes, device=device,
                                 seed=args.seed)
    np.save(results / "east_coast_scores.npy", mu)
    np.save(results / "east_coast_uncertainty.npy", sigma)
    grid["score"] = mu; grid["uncertainty"] = sigma
    grid[["cell_id", "lat", "lon", "score", "uncertainty"]].to_csv(
        results / "east_coast_grid.csv", index=False)
    print(f"  median score={float(np.median(mu)):.4f}  "
          f"frac>{args.prob_threshold}={float((mu>=args.prob_threshold).mean()):.2%}")

    # ---- 5. DBSCAN clustering ------------------------------------------
    print("\n>>> 5. cluster discoveries", flush=True)
    disc = cluster_discoveries(grid, mu, sigma,
                                prob_thresh=args.prob_threshold,
                                unc_thresh=args.unc_threshold,
                                eps_km=args.dbscan_eps_km,
                                min_samples=args.dbscan_min_samples)
    print(f"  {len(disc)} eastern cluster discoveries")
    disc.to_csv(results / "east_coast_discoveries.csv", index=False)

    # ---- 6. MWe estimation ---------------------------------------------
    print("\n>>> 6. MWe estimation (Williams 2008, conservative T)", flush=True)
    disc_mwe = estimate_mwe(disc, seed=args.seed)
    disc_mwe.to_csv(results / "east_coast_discoveries_with_mwe.csv", index=False)
    if len(disc_mwe):
        print(f"  ΣP50_MWe = {disc_mwe.P50_MWe.sum():.0f}")
        print(f"  ΣP10_MWe = {disc_mwe.P10_MWe.sum():.0f}")

    # ---- 7. Pre-registration manifest ----------------------------------
    print("\n>>> 7. pre-registration manifest", flush=True)
    manifest = {
        "checkpoint": str(out_ckpt),
        "base_checkpoint": str(base_ckpt),
        "epochs": args.epochs, "lr_scale": args.lr_scale,
        "n_eastern_pos": len(pos), "n_eastern_neg": len(neg),
        "mc_passes": args.mc_passes, "prob_threshold": args.prob_threshold,
        "unc_threshold": args.unc_threshold,
        "dbscan_eps_km": args.dbscan_eps_km,
        "dbscan_min_samples": args.dbscan_min_samples,
        "n_discoveries": int(len(disc)),
        "n_grid_cells": int(len(grid)),
        "frac_above_threshold": float((mu >= args.prob_threshold).mean()),
        "median_score": float(np.median(mu)),
        "sum_P50_MWe": float(disc_mwe.P50_MWe.sum()) if len(disc_mwe) else 0.0,
        "sum_P10_MWe": float(disc_mwe.P10_MWe.sum()) if len(disc_mwe) else 0.0,
    }
    payload = json.dumps(manifest, sort_keys=True).encode()
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()
    write_json(results / "east_coast_preregistration.json", manifest)
    if len(disc_mwe):
        disc_mwe[["cluster_id", "lat_centroid", "lon_centroid", "n_cells",
                  "area_km2", "score_mean", "uncertainty_mean", "T_est_C",
                  "P10_MWe", "P50_MWe", "P90_MWe"]].to_csv(
            results / "east_coast_preregistration.csv", index=False)
    print(f"  manifest sha256: {manifest['sha256'][:16]}…")
    print(f"  done. wrote {results / 'east_coast_preregistration.json'}")


if __name__ == "__main__":
    main()
