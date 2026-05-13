"""Score post-2008 NV permit sites against the trained model.

For each held-out permit cell, compute:
  - the predicted favourability percentile (across all 235K continental cells)
  - whether it lands in top {1, 5, 10, 25}%
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import GeoProspectDataset, make_loader
from src.models.geoprospectnet import GeoProspectNet


def main(config_path: str = "configs/cpu_max.yaml",
         checkpoint: str | None = None,
         output_csv: str = "outputs/results/temporal_holdout.csv") -> None:
    cfg = yaml.safe_load(open(ROOT / config_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint is None:
        # Prefer the headline cpu_max checkpoint if present, else fall back
        # to whatever random_seed* checkpoint exists.
        cand = ROOT / "outputs/checkpoints/random_cpu_max_seed42.pt"
        if cand.exists():
            ckpt = cand
        else:
            ckpts = list((ROOT / "outputs/checkpoints").glob("random_seed*_best.pt"))
            if not ckpts:
                ckpts = sorted((ROOT / "outputs/checkpoints").glob("*_best.pt"),
                                key=lambda p: p.stat().st_mtime)
            ckpt = ckpts[-1]
    else:
        ckpt = Path(checkpoint)
    print(f"loading checkpoint: {ckpt}")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    # Use the config the checkpoint was trained with if it stored one — that's
    # the only way embed_dim / fused_dim line up across configs.
    if isinstance(ck, dict) and "config" in ck:
        cfg = ck["config"]
    model = GeoProspectNet(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    grid = pd.read_csv(ROOT / "data/processed/grid_coordinates.csv")
    ds = GeoProspectDataset(ROOT / "data/processed",
                             indices=np.arange(len(grid)),
                             use_thermal=cfg["modalities"]["use_thermal"])
    loader = make_loader(ds, batch_size=512, shuffle=False, num_workers=0)
    print(f"scoring {len(grid):,} cells ...")
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            scores.append(torch.sigmoid(out["logits"]).cpu().numpy())
    scores_np = np.concatenate(scores)
    print(f"score range: {scores_np.min():.4f} to {scores_np.max():.4f}, "
          f"mean={scores_np.mean():.4f}")

    holdout = pd.read_csv(ROOT / "data/raw/labels/nv_permits_holdout.csv")
    novel = holdout[holdout.is_novel].reset_index(drop=True)
    print(f"novel sites: {len(novel)}")

    lat0 = float(grid.lat.mean()); cos_lat0 = float(np.cos(np.radians(lat0)))
    gxy = np.column_stack([grid.lon * 111.32 * cos_lat0, grid.lat * 111.32])
    tree = cKDTree(gxy)

    rows = []
    for _, row in novel.iterrows():
        px = row.lon * 111.32 * cos_lat0
        py = row.lat * 111.32
        _, idx = tree.query([px, py], k=1)
        p = float(scores_np[idx])
        pct = float(100.0 * (scores_np < p).mean())
        rows.append({
            "permit": row.get("permit", ""),
            "well": row.get("well", ""),
            "operator": str(row.get("operator", ""))[:30],
            "lat": row.lat, "lon": row.lon,
            "predicted_p": p,
            "percentile": pct,
            "in_top_1pct": pct >= 99.0,
            "in_top_5pct": pct >= 95.0,
            "in_top_10pct": pct >= 90.0,
            "in_top_25pct": pct >= 75.0,
        })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / output_csv, index=False)
    print()
    print(out.sort_values("percentile", ascending=False).to_string(index=False))
    print()
    print(f"Aggregate temporal-holdout capture rates:")
    print(f"  in top  1%: {out.in_top_1pct.mean():.1%}  ({out.in_top_1pct.sum()}/{len(out)})")
    print(f"  in top  5%: {out.in_top_5pct.mean():.1%}  ({out.in_top_5pct.sum()}/{len(out)})")
    print(f"  in top 10%: {out.in_top_10pct.mean():.1%}  ({out.in_top_10pct.sum()}/{len(out)})")
    print(f"  in top 25%: {out.in_top_25pct.mean():.1%}  ({out.in_top_25pct.sum()}/{len(out)})")


if __name__ == "__main__":
    main()
