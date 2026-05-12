"""Calibration analysis: reliability diagram, Brier score, Expected Calibration Error.

Uses cached continental scores (e.g. outputs/results/scores_cpu_max.npy) and
the labeled cells from train_mask + labels.npy. Bins predicted probability
into 10 equal-width bins and reports observed positive frequency per bin.

Outputs:
    outputs/results/calibration_table.csv
    outputs/results/calibration_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def reliability(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    df_rows = []
    ece = 0.0
    n_total = len(y_true)
    for i in range(n_bins):
        mask = (p_pred >= bins[i]) & (p_pred < bins[i + 1] if i + 1 < n_bins else p_pred <= bins[i + 1])
        n = int(mask.sum())
        if n == 0:
            df_rows.append(dict(bin_low=bins[i], bin_high=bins[i + 1], n=0,
                                mean_pred=np.nan, obs_freq=np.nan))
            continue
        mp = float(p_pred[mask].mean())
        of = float(y_true[mask].mean())
        df_rows.append(dict(bin_low=bins[i], bin_high=bins[i + 1], n=n,
                            mean_pred=mp, obs_freq=of))
        ece += (n / n_total) * abs(mp - of)
    return pd.DataFrame(df_rows), float(ece)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_max.yaml")
    parser.add_argument("--scores", default="outputs/results/scores_cpu_max.npy")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    processed = ROOT / cfg["paths"]["processed_dir"]

    scores = np.load(ROOT / args.scores)
    labels = np.load(processed / "labels.npy")
    train_mask = np.load(processed / "train_mask.npy")

    labeled = np.flatnonzero(train_mask)
    y = labels[labeled].astype(np.float32)
    p = scores[labeled].astype(np.float32)

    # Brier score
    brier = float(((p - y) ** 2).mean())
    # ECE
    df, ece = reliability(y, p, n_bins=10)
    out_csv = ROOT / "outputs/results/calibration_table.csv"
    df.to_csv(out_csv, index=False)

    summary = dict(
        brier_score=brier,
        expected_calibration_error=ece,
        n_labeled=int(len(y)),
        n_positives=int(y.sum()),
        n_negatives=int(len(y) - y.sum()),
    )
    out_json = ROOT / "outputs/results/calibration_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print(df.to_string(index=False, float_format="%.4f"))
    print(f"\nBrier score:                {brier:.4f}  (lower is better; 0 = perfect)")
    print(f"Expected Calibration Error: {ece:.4f}  (lower is better)")
    print(f"\nWrote {out_csv}, {out_json}")


if __name__ == "__main__":
    main()
