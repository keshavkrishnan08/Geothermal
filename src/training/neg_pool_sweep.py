"""Negative-pool-size sweep for cpu_margin.

For each ratio: rebuild labels.npy with that negative_pos_ratio, train
cpu_margin from scratch, evaluate, snapshot the checkpoint. Restore the
original labels.npy at the end so downstream artifacts remain consistent.

Outputs ``outputs/results/neg_pool_sweep.csv``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
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

    ho = pd.read_csv(ROOT / "data/raw/labels/nv_permits_holdout.csv")
    novel = ho[ho.is_novel].reset_index(drop=True) if "is_novel" in ho.columns else ho
    tree = cKDTree(gxy)
    ho_idx = []
    for _, r in novel.iterrows():
        _, i = tree.query([r.lon * 111.32 * cos_lat0, r.lat * 111.32], k=1)
        ho_idx.append(i)
    ho_pcts = np.array([100.0 * (scores < scores[i]).mean() for i in ho_idx])
    return dict(
        pos_pct=pos_pct, nc3_pct=nc3_pct, gap=pos_pct - nc3_pct,
        holdout_mean_pct=float(ho_pcts.mean()),
        holdout_top10_capture=float((ho_pcts >= 90).mean()),
        n_negatives=int((labels == 0).sum()),
    )


def rebuild_labels(base_cfg_path: str, neg_ratio: int):
    """Edit cpu_margin.yaml in place to set neg_pos_ratio, then rebuild labels."""
    cfg = yaml.safe_load(open(base_cfg_path))
    cfg["labels"]["negative_pos_ratio"] = neg_ratio
    tmp = Path(tempfile.mkdtemp()) / "tmp_neg_sweep.yaml"
    tmp.write_text(yaml.safe_dump(cfg))
    subprocess.run(["python", "-m", "src.data.build_labels", "--config", str(tmp)],
                   cwd=str(ROOT), check=True)
    return tmp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/cpu_margin.yaml")
    parser.add_argument("--ratios", type=int, nargs="+", default=[3, 5, 10, 20])
    args = parser.parse_args()

    # Snapshot the current labels so we can restore afterwards
    labels_snapshot = ROOT / "data/processed/labels_neg_pool_snapshot.npy"
    train_snapshot = ROOT / "data/processed/train_mask_neg_pool_snapshot.npy"
    shutil.copy(ROOT / "data/processed/labels.npy", labels_snapshot)
    shutil.copy(ROOT / "data/processed/train_mask.npy", train_snapshot)
    print(f"snapshotted labels.npy and train_mask.npy")

    rows = []
    try:
        for ratio in args.ratios:
            named_ckpt = ROOT / f"outputs/checkpoints/random_cpu_margin_neg{ratio}_seed42.pt"
            if named_ckpt.exists():
                print(f"\n=== ratio {ratio}: already trained, evaluating only ===")
                cfg = yaml.safe_load(open(args.base_config))
                metrics = _eval(named_ckpt, cfg)
            else:
                print(f"\n=== ratio {ratio}: rebuilding labels and training ===")
                tmp_cfg = rebuild_labels(args.base_config, ratio)
                subprocess.run([
                    "python", "-m", "src.training.train",
                    "--config", str(tmp_cfg), "--random", "--seed", "42",
                ], cwd=str(ROOT), check=True)
                per_seed = ROOT / "outputs/checkpoints/random_seed42_best.pt"
                shutil.copy(per_seed, named_ckpt)
                cfg = yaml.safe_load(open(tmp_cfg))
                metrics = _eval(named_ckpt, cfg)
            metrics["neg_pos_ratio"] = ratio
            rows.append(metrics)
            print(f"  ratio={ratio}  n_neg={metrics['n_negatives']}  "
                  f"pos={metrics['pos_pct']:.1f}  nc3={metrics['nc3_pct']:.1f}  "
                  f"gap={metrics['gap']:.1f}  ho={metrics['holdout_mean_pct']:.1f}")
    finally:
        # Restore labels regardless of success
        shutil.copy(labels_snapshot, ROOT / "data/processed/labels.npy")
        shutil.copy(train_snapshot, ROOT / "data/processed/train_mask.npy")
        labels_snapshot.unlink(missing_ok=True)
        train_snapshot.unlink(missing_ok=True)
        print("\nrestored original labels.npy and train_mask.npy")

    df = pd.DataFrame(rows).sort_values("neg_pos_ratio")
    out = ROOT / "outputs/results/neg_pool_sweep.csv"
    df.to_csv(out, index=False)
    print("\n=== NEGATIVE-POOL SIZE SWEEP ===")
    print(df.to_string(index=False, float_format="%.2f"))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
