"""Sweep training configs and rank them by positive-vs-negative percentile gap.

For each config:
  1. Re-build labels with config-specified neg/pos ratio (only if changed)
  2. Train the model (random 70/15/15 split)
  3. Run continental inference
  4. Compute mean percentile for: known positives, post-2008 hold-out, NC1, NC3
  5. Rank by GAP = mean(positives) - mean(NC3)

Writes outputs/results/separation_sweep.csv.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _continental_scores(ckpt_path: Path, cfg: Dict, batch_size: int = 512) -> np.ndarray:
    import sys; sys.path.insert(0, str(ROOT))
    from src.data.dataset import GeoProspectDataset, make_loader
    from src.models.geoprospectnet import GeoProspectNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GeoProspectNet(cfg).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    n_total = int(np.load(ROOT / "data/processed/labels.npy", mmap_mode="r").shape[0])
    ds = GeoProspectDataset(ROOT / "data/processed", indices=np.arange(n_total),
                            use_thermal=cfg["modalities"]["use_thermal"])
    loader = make_loader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            o = model(batch)
            out.append(torch.sigmoid(o["logits"]).cpu().numpy())
    return np.concatenate(out)


def _percentiles(scores: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.array([100.0 * (scores < scores[i]).mean() for i in idx])


def _build_neg_controls(grid: pd.DataFrame, fields: pd.DataFrame,
                        masks: np.ndarray, n: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    lat0 = float(grid.lat.mean()); cos_lat0 = float(np.cos(np.radians(lat0)))
    fxy = np.column_stack([fields.lon * 111.32 * cos_lat0, fields.lat * 111.32])
    gxy = np.column_stack([grid.lon * 111.32 * cos_lat0, grid.lat * 111.32])
    from scipy.spatial import cKDTree
    d_to_field, _ = cKDTree(fxy).query(gxy, k=1)

    nc1 = grid.lat.between(36, 40) & grid.lon.between(-111, -108) & (d_to_field > 200)
    nc3 = (d_to_field > 100) & (~masks[:, 1])

    def sample(m):
        idx = np.flatnonzero(m.values if hasattr(m, "values") else m)
        return rng.choice(idx, size=min(n, len(idx)), replace=False) if len(idx) else np.array([], dtype=int)

    return {"NC1": sample(nc1), "NC3": sample(nc3),
            "random": rng.choice(len(grid), size=n, replace=False)}


def evaluate(ckpt_path: Path, config_path: str, holdout_path: Path) -> Dict:
    cfg = yaml.safe_load(open(config_path))
    scores = _continental_scores(ckpt_path, cfg)
    grid = pd.read_csv(ROOT / "data/processed/grid_coordinates.csv")
    fields = pd.read_csv(ROOT / "data/metadata/known_fields_details.csv")
    masks = np.load(ROOT / "data/processed/modality_masks.npy")
    labels = np.load(ROOT / "data/processed/labels.npy")

    pos_idx = np.flatnonzero(labels == 1)
    pos_pcts = _percentiles(scores, pos_idx)

    nc = _build_neg_controls(grid, fields, masks)
    nc_pcts = {k: _percentiles(scores, v) if len(v) else np.array([]) for k, v in nc.items()}

    # Hold-out
    hp = pd.read_csv(holdout_path)
    novel = hp[hp.is_novel].reset_index(drop=True) if "is_novel" in hp.columns else hp
    lat0 = float(grid.lat.mean()); cos_lat0 = float(np.cos(np.radians(lat0)))
    from scipy.spatial import cKDTree
    gxy = np.column_stack([grid.lon * 111.32 * cos_lat0, grid.lat * 111.32])
    tree = cKDTree(gxy)
    h_idx = []
    for _, r in novel.iterrows():
        _, i = tree.query([r.lon * 111.32 * cos_lat0, r.lat * 111.32], k=1)
        h_idx.append(i)
    holdout_pcts = _percentiles(scores, np.array(h_idx))

    summary = {
        "score_median": float(np.median(scores)),
        "score_frac_above_50": float((scores > 0.5).mean()),
        "pos_mean_pct": float(pos_pcts.mean()),
        "pos_median_pct": float(np.median(pos_pcts)),
        "holdout_mean_pct": float(holdout_pcts.mean()) if len(holdout_pcts) else float("nan"),
        "holdout_capture_top10": float((holdout_pcts > 90).mean()) if len(holdout_pcts) else float("nan"),
        "nc1_mean_pct": float(nc_pcts["NC1"].mean()) if len(nc_pcts["NC1"]) else float("nan"),
        "nc3_mean_pct": float(nc_pcts["NC3"].mean()) if len(nc_pcts["NC3"]) else float("nan"),
        "random_mean_pct": float(nc_pcts["random"].mean()),
        # SEPARATION GAP — the headline metric to maximise
        "separation_gap": float(pos_pcts.mean() - nc_pcts["NC3"].mean()) if len(nc_pcts["NC3"]) else float("nan"),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+",
                        default=["configs/cpu_calibrated.yaml",
                                 "configs/cpu_tuned.yaml",
                                 "configs/cpu_margin.yaml"])
    parser.add_argument("--holdout", default="data/raw/labels/nv_permits_holdout.csv")
    args = parser.parse_args()

    rows = []
    for cfg_path in args.configs:
        name = Path(cfg_path).stem
        print(f"\n=== {name} ===")
        # Train (random split)
        ckpt = ROOT / f"outputs/checkpoints/random_seed42_best.pt"
        ckpt.unlink(missing_ok=True)
        subprocess.run(["python", "-m", "src.training.train", "--config", cfg_path,
                        "--random", "--seed", "42"], cwd=str(ROOT), check=True)
        # Move so the next iteration doesn't overwrite
        new_ckpt = ROOT / f"outputs/checkpoints/random_{name}_seed42.pt"
        shutil.copy(ckpt, new_ckpt)
        # Evaluate
        try:
            row = evaluate(ckpt, cfg_path, ROOT / args.holdout)
            row["config"] = name; row["ckpt"] = str(new_ckpt.name)
            rows.append(row)
            print(f"  pos={row['pos_mean_pct']:.1f}  nc3={row['nc3_mean_pct']:.1f}  gap={row['separation_gap']:.1f}")
        except Exception as e:
            print(f"  eval failed: {e}")

    df = pd.DataFrame(rows).sort_values("separation_gap", ascending=False)
    out = ROOT / "outputs/results/separation_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\n=== ranked by separation_gap ===")
    print(df[["config", "pos_mean_pct", "holdout_mean_pct", "random_mean_pct",
              "nc1_mean_pct", "nc3_mean_pct", "separation_gap",
              "score_frac_above_50"]].to_string(index=False, float_format="%.2f"))
    print(f"\nwinner: {df.iloc[0]['config']}  gap={df.iloc[0]['separation_gap']:.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
