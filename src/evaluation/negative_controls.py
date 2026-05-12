"""Negative-control evaluation — does the model correctly say NO to obviously
non-prospective places?

Reviewers will demand this. A "favourability" model that gives 0.99 to deep
craton interiors is broken regardless of what it does at known fields.

We build three negative-control sets and compare their predicted percentiles
to (a) known fields, (b) the post-2008 NV permit hold-out, and (c) random
cells in the western US.

Negative controls:
    NC1 — Colorado Plateau interior (high elevation craton, no Quaternary
          faulting, no Holocene volcanism within 200 km)
    NC2 — Eastern Great Basin sedimentary basins far from any active fault
    NC3 — Random "deep" cells: > 100 km from any positive label, > 50 km
          from any thermal spring, low heat flow
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree

from src.data.dataset import GeoProspectDataset, make_loader
from src.models.geoprospectnet import GeoProspectNet

ROOT = Path(__file__).resolve().parents[2]


def _continental_scores(model, processed: Path, use_thermal: bool, device: str,
                         batch_size: int = 512) -> np.ndarray:
    n_total = int(np.load(processed / "labels.npy", mmap_mode="r").shape[0])
    ds = GeoProspectDataset(processed, indices=np.arange(n_total), use_thermal=use_thermal)
    loader = make_loader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    scores = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            scores.append(torch.sigmoid(out["logits"]).cpu().numpy())
    return np.concatenate(scores)


def _build_neg_controls(grid: pd.DataFrame, fields: pd.DataFrame,
                        masks: np.ndarray, n_each: int = 50,
                        seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    lat0 = float(grid.lat.mean()); cos_lat0 = float(np.cos(np.radians(lat0)))

    # Distance to nearest known field
    fxy = np.column_stack([fields.lon * 111.32 * cos_lat0, fields.lat * 111.32])
    gxy = np.column_stack([grid.lon * 111.32 * cos_lat0, grid.lat * 111.32])
    ftree = cKDTree(fxy)
    d_to_field, _ = ftree.query(gxy, k=1)

    # NC1 — Colorado Plateau interior: lat 36-40, lon -111 to -108, > 200 km from any field
    nc1_mask = (
        grid.lat.between(36, 40) & grid.lon.between(-111, -108)
        & (d_to_field > 200)
    )
    # NC2 — Eastern Great Basin sed basins far from active faulting:
    #       lat 38-42, lon -116 to -113, > 100 km from any field, no spring coverage (mask 1=False)
    nc2_mask = (
        grid.lat.between(38, 42) & grid.lon.between(-116, -113)
        & (d_to_field > 100) & (~masks[:, 1])  # geochem mask off ⇒ no springs nearby
    )
    # NC3 — Random "deep" cells: > 100 km from any positive, no spring coverage
    nc3_mask = (d_to_field > 100) & (~masks[:, 1])

    def sample(mask, n, label):
        idx = np.flatnonzero(mask.values if hasattr(mask, "values") else mask)
        if len(idx) == 0:
            return pd.DataFrame()
        chosen = rng.choice(idx, size=min(n, len(idx)), replace=False)
        df = grid.iloc[chosen].copy()
        df["control"] = label
        df["nearest_field_km"] = d_to_field[chosen]
        return df

    return {
        "NC1_colorado_plateau": sample(nc1_mask, n_each, "NC1_colorado_plateau"),
        "NC2_eastern_basin": sample(nc2_mask, n_each, "NC2_eastern_basin"),
        "NC3_random_deep": sample(nc3_mask, n_each, "NC3_random_deep"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_calibrated.yaml")
    parser.add_argument("--ckpt", default=None,
                        help="Override checkpoint path; default uses random_seed42_best.pt")
    parser.add_argument("--n_each", type=int, default=100)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed = ROOT / cfg["paths"]["processed_dir"]

    ckpt_path = args.ckpt or "outputs/checkpoints/random_seed42_best.pt"
    ck = torch.load(ROOT / ckpt_path, map_location=device, weights_only=False)
    model = GeoProspectNet(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    print(f"loaded {ckpt_path}")

    print("scoring continental grid ...")
    scores = _continental_scores(model, processed, cfg["modalities"]["use_thermal"], device)

    grid = pd.read_csv(processed / "grid_coordinates.csv")
    fields = pd.read_csv(ROOT / cfg["paths"]["metadata_dir"] / "known_fields_details.csv")
    masks = np.load(processed / "modality_masks.npy")

    controls = _build_neg_controls(grid, fields, masks, n_each=args.n_each)

    rows = []
    for name, df in controls.items():
        if df.empty: continue
        cell_ids = df["cell_id"].values
        ps = scores[cell_ids]
        pcts = np.array([100.0 * (scores < p).mean() for p in ps])
        rows.append({
            "cohort": name,
            "n": len(ps),
            "mean_p": float(ps.mean()),
            "median_p": float(np.median(ps)),
            "mean_percentile": float(pcts.mean()),
            "median_percentile": float(np.median(pcts)),
            "frac_above_50pct": float((pcts > 50).mean()),
            "frac_above_90pct": float((pcts > 90).mean()),
        })

    # Random western-US baseline
    rng = np.random.default_rng(42)
    rand_idx = rng.choice(len(grid), size=args.n_each, replace=False)
    ps = scores[rand_idx]
    pcts = np.array([100.0 * (scores < p).mean() for p in ps])
    rows.append({
        "cohort": "random_western_us",
        "n": len(ps),
        "mean_p": float(ps.mean()),
        "median_p": float(np.median(ps)),
        "mean_percentile": float(pcts.mean()),
        "median_percentile": float(np.median(pcts)),
        "frac_above_50pct": float((pcts > 50).mean()),
        "frac_above_90pct": float((pcts > 90).mean()),
    })

    # Known positive labels (sanity reference)
    labels = np.load(processed / "labels.npy")
    pos_idx = np.flatnonzero(labels == 1)
    ps = scores[pos_idx]
    pcts = np.array([100.0 * (scores < p).mean() for p in ps])
    rows.append({
        "cohort": "known_positives",
        "n": len(ps),
        "mean_p": float(ps.mean()),
        "median_p": float(np.median(ps)),
        "mean_percentile": float(pcts.mean()),
        "median_percentile": float(np.median(pcts)),
        "frac_above_50pct": float((pcts > 50).mean()),
        "frac_above_90pct": float((pcts > 90).mean()),
    })

    # Temporal hold-out (post-2008 NV permits)
    holdout_path = ROOT / "outputs/results/temporal_holdout.csv"
    if holdout_path.exists():
        h = pd.read_csv(holdout_path)
        rows.append({
            "cohort": "temporal_holdout_post2008",
            "n": len(h),
            "mean_p": float(h.predicted_p.mean()),
            "median_p": float(h.predicted_p.median()),
            "mean_percentile": float(h.percentile.mean()),
            "median_percentile": float(h.percentile.median()),
            "frac_above_50pct": float((h.percentile > 50).mean()),
            "frac_above_90pct": float((h.percentile > 90).mean()),
        })

    df = pd.DataFrame(rows)
    out = ROOT / "outputs/results/negative_controls.csv"
    df.to_csv(out, index=False)
    print()
    print(df.to_string(index=False, float_format="%.3f"))
    print()
    print(f"Wrote {out}")
    print()
    print("Interpretation:")
    print("  HEALTHY model: known_positives mean_pct >> NC1/NC2/NC3 mean_pct,")
    print("                 NC mean_pct ~ 30-50, random_western_us ~ 50.")
    print("  BROKEN  model: NC mean_pct also > 80 (model predicts positive everywhere).")


if __name__ == "__main__":
    main()
