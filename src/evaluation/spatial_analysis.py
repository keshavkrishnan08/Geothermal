"""Spatial autocorrelation (Moran's I) and α sensitivity for spatial smoothing."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.train import train_one_split
from src.training.utils import device_auto, load_config

ROOT = Path(__file__).resolve().parents[2]


def morans_i(values: np.ndarray, neighbor_indices: np.ndarray) -> float:
    """Compute Moran's I given values at every cell and a k-NN neighbor table.

    The neighbor weight is uniform: w_ij = 1/k if j is one of i's k neighbors.
    """
    n = len(values)
    k = neighbor_indices.shape[1]
    x = values - values.mean()
    var = np.sum(x ** 2)
    if var == 0:
        return float("nan")

    num = 0.0
    for j in range(k):
        num += np.sum(x * x[neighbor_indices[:, j]])
    W = n * k * (1.0 / k)  # = n
    I = (n / W) * (num / var)
    return float(I)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--alphas", nargs="*", type=float,
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    processed = ROOT / cfg["paths"]["processed_dir"]
    folds = json.load(open(ROOT / cfg["paths"]["splits_dir"] / "lof_cv_folds.json"))
    folds = folds[: args.n_folds]

    rows = []
    for alpha in args.alphas:
        run = copy.deepcopy(cfg)
        run["model"]["spatial_alpha"] = alpha
        aurocs = []
        for fold in folds:
            r, _ = train_one_split(run, np.array(fold["train_idx"]),
                                   np.array(fold["test_idx"]),
                                   seed=42, device=device,
                                   log_prefix=f"alpha{alpha}_fold{fold['field_id']}_")
            aurocs.append(r["best"]["auroc"])
        rows.append({"alpha": alpha, "auroc_mean": float(np.nanmean(aurocs)),
                     "auroc_std": float(np.nanstd(aurocs))})
        print(f"alpha={alpha:.2f}  AUROC={rows[-1]['auroc_mean']:.4f} "
              f"± {rows[-1]['auroc_std']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/results/alpha_sensitivity.csv", index=False)

    # Moran's I on saved prospectivity map (if it exists)
    mean_p_path = ROOT / "outputs/results/prospectivity_mean.npy"
    neighbors_path = processed / "neighbor_indices.npy"
    if mean_p_path.exists() and neighbors_path.exists():
        values = np.load(mean_p_path)
        neighbors = np.load(neighbors_path)
        I = morans_i(values, neighbors)
        with open(ROOT / "outputs/results/morans_i.json", "w") as f:
            json.dump({"morans_i": I, "n": int(len(values))}, f, indent=2)
        print(f"Moran's I on prospectivity map: {I:.4f}")


if __name__ == "__main__":
    main()
