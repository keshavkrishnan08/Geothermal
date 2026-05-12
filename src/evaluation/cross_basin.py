"""Cross-basin transfer experiment.

Train on one tectonic province, evaluate on another. Tests whether the
learnt prospectivity signal generalises across geological settings — a
question no published geothermal-ML study has answered.

Default protocol:
    Train pool:  Basin and Range positives + matched negatives
    Test pool:   {Cascades, Snake_River_Plain} positives + matched negatives
We report top-{1,5,10}% capture rate on each test province.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import GeoProspectDataset, make_loader
from src.training.train import train_one_split, metrics_at_threshold
from src.training.utils import device_auto, load_config

ROOT = Path(__file__).resolve().parents[2]


def _split_by_province(grid_df: pd.DataFrame, train_provinces: list,
                       test_provinces: list, train_mask: np.ndarray,
                       labels: np.ndarray):
    """Return (train_idx, test_idx) restricted to the given provinces."""
    in_train_prov = grid_df["province"].isin(train_provinces).values
    in_test_prov = grid_df["province"].isin(test_provinces).values
    train_idx = np.flatnonzero(train_mask & in_train_prov)
    test_idx = np.flatnonzero(train_mask & in_test_prov)
    return train_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train_provinces", nargs="+",
                        default=["Basin_and_Range"])
    parser.add_argument("--test_provinces", nargs="+",
                        default=["Cascades", "Snake_River_Plain"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    processed = ROOT / cfg["paths"]["processed_dir"]

    grid_df = pd.read_csv(processed / "grid_coordinates.csv")
    if "province" not in grid_df.columns:
        prov_csv = ROOT / cfg["paths"]["metadata_dir"] / "tectonic_provinces.csv"
        prov = pd.read_csv(prov_csv)
        grid_df = grid_df.merge(prov[["cell_id", "province"]], on="cell_id", how="left")

    train_mask = np.load(processed / "train_mask.npy")
    labels = np.load(processed / "labels.npy")

    train_idx, test_idx = _split_by_province(
        grid_df, args.train_provinces, args.test_provinces, train_mask, labels,
    )
    print(f"Train: {len(train_idx)} cells in {args.train_provinces}  "
          f"({int(labels[train_idx].sum())} pos)")
    print(f"Test:  {len(test_idx)} cells in {args.test_provinces}  "
          f"({int(labels[test_idx].sum())} pos)")

    if len(train_idx) < 100 or len(test_idx) < 50:
        print("[warn] insufficient cells in one of the provinces — aborting")
        return

    r, _ = train_one_split(
        cfg, train_idx, test_idx, seed=args.seed, device=device,
        log_prefix=f"crossbasin_{'_'.join(args.train_provinces)}_to_{'_'.join(args.test_provinces)}_",
    )
    out = ROOT / "outputs/results/cross_basin_transfer.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_provinces": args.train_provinces,
        "test_provinces": args.test_provinces,
        "n_train": int(len(train_idx)),
        "n_train_pos": int(labels[train_idx].sum()),
        "n_test": int(len(test_idx)),
        "n_test_pos": int(labels[test_idx].sum()),
        "metrics": r["best"],
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(json.dumps(payload, indent=2, default=float))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
