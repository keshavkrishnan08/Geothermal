"""Train cpu_margin with multiple seeds and report mean ± std.

Saves outputs/results/multi_seed_sensitivity.csv.

Each fold:
  1. Train with --seed=s
  2. Continental-infer
  3. Compute separation metrics
  4. Save checkpoint to random_cpu_margin_seed{s}.pt
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]


def _eval(ckpt_path: Path, cfg) -> dict:
    import sys; sys.path.insert(0, str(ROOT))
    from src.data.dataset import GeoProspectDataset, make_loader
    from src.models.geoprospectnet import GeoProspectNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GeoProspectNet(cfg).to(device); model.load_state_dict(ck["state_dict"]); model.eval()

    n = int(np.load(ROOT / "data/processed/labels.npy", mmap_mode="r").shape[0])
    ds = GeoProspectDataset(ROOT / "data/processed", indices=np.arange(n),
                            use_thermal=cfg["modalities"]["use_thermal"])
    loader = make_loader(ds, batch_size=512, shuffle=False, num_workers=0)
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            scores.append(torch.sigmoid(model(batch)["logits"]).cpu().numpy())
    scores = np.concatenate(scores)

    grid = pd.read_csv(ROOT / "data/processed/grid_coordinates.csv")
    fields = pd.read_csv(ROOT / "data/metadata/known_fields_details.csv")
    masks = np.load(ROOT / "data/processed/modality_masks.npy")
    labels = np.load(ROOT / "data/processed/labels.npy")

    lat0 = float(grid.lat.mean()); cos_lat0 = float(np.cos(np.radians(lat0)))
    gxy = np.column_stack([grid.lon * 111.32 * cos_lat0, grid.lat * 111.32])
    fxy = np.column_stack([fields.lon * 111.32 * cos_lat0, fields.lat * 111.32])
    d, _ = cKDTree(fxy).query(gxy, k=1)
    nc3 = (d > 100) & (~masks[:, 1])
    rng = np.random.default_rng(42)
    nc3_idx = rng.choice(np.flatnonzero(nc3), size=100, replace=False)
    pos_idx = np.flatnonzero(labels == 1)

    pos_pct = float(np.mean([100.0 * (scores < scores[i]).mean() for i in pos_idx]))
    nc3_pct = float(np.mean([100.0 * (scores < scores[i]).mean() for i in nc3_idx]))

    # Hold-out
    ho = pd.read_csv(ROOT / "data/raw/labels/nv_permits_holdout.csv")
    novel = ho[ho.is_novel].reset_index(drop=True) if "is_novel" in ho.columns else ho
    tree = cKDTree(gxy)
    ho_idx = []
    for _, r in novel.iterrows():
        _, i = tree.query([r.lon * 111.32 * cos_lat0, r.lat * 111.32], k=1)
        ho_idx.append(i)
    ho_idx = np.array(ho_idx)
    ho_pcts = np.array([100.0 * (scores < scores[i]).mean() for i in ho_idx])

    return dict(
        pos_pct=pos_pct, nc3_pct=nc3_pct, gap=pos_pct - nc3_pct,
        holdout_mean_pct=float(ho_pcts.mean()),
        holdout_top10_capture=float((ho_pcts >= 90).mean()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_margin.yaml")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 7, 13, 21, 99])
    args = parser.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    cfg_name = Path(args.config).stem

    rows = []
    for s in args.seeds:
        named_ckpt = ROOT / f"outputs/checkpoints/random_{cfg_name}_seed{s}.pt"
        # train.py saves with prefix from --random + seed → random_seed{s}_best.pt
        per_seed_ckpt = ROOT / f"outputs/checkpoints/random_seed{s}_best.pt"
        if not named_ckpt.exists():
            if not per_seed_ckpt.exists():
                print(f"\n=== seed {s}: training ===", flush=True)
                subprocess.run([
                    "python", "-m", "src.training.train",
                    "--config", args.config, "--random", "--seed", str(s),
                ], cwd=str(ROOT), check=True)
            import shutil
            shutil.copy(per_seed_ckpt, named_ckpt)
            print(f"  saved {named_ckpt.name}", flush=True)
        print(f"--- evaluating seed {s} ---", flush=True)
        metrics = _eval(named_ckpt, cfg)
        metrics["seed"] = s
        rows.append(metrics)
        print(f"  pos={metrics['pos_pct']:.1f} nc3={metrics['nc3_pct']:.1f} "
              f"gap={metrics['gap']:.1f} ho={metrics['holdout_mean_pct']:.1f}",
              flush=True)

    df = pd.DataFrame(rows)
    print("\n=== MULTI-SEED SENSITIVITY ===")
    print(df.to_string(index=False, float_format="%.2f"))
    print()
    print("Aggregate (mean ± std):")
    for col in ["pos_pct", "nc3_pct", "gap", "holdout_mean_pct", "holdout_top10_capture"]:
        mu = float(df[col].mean()); sd = float(df[col].std())
        print(f"  {col:25s} = {mu:.3f} ± {sd:.3f}")
    out = ROOT / "outputs/results/multi_seed_sensitivity.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
