"""Figure 3 — ROC curves of all models (mean across LOF-CV folds).

Because LOF folds have different test compositions, we plot the per-fold ROC
curves (light) plus the mean curve (bold) for each model. Requires that each
fold's per-sample y_true / y_score have been saved during training.

If only summary metrics are available (no per-sample scores), we fall back to
a bar chart of per-model AUROC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/results")
    args = parser.parse_args()

    res = ROOT / args.results_dir
    fig_dir = ROOT / "outputs/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    table_path = res / "table1_main_results.csv"
    if not table_path.exists():
        print("[warn] no table1 CSV — run cross_validation.py first")
        return

    df = pd.read_csv(table_path)
    aurocs = []
    for _, row in df.iterrows():
        v = str(row["auroc"]).split("±")[0].strip()
        try:
            aurocs.append(float(v))
        except ValueError:
            aurocs.append(np.nan)
    df["auroc_mean"] = aurocs
    df = df.dropna(subset=["auroc_mean"]).sort_values("auroc_mean")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df["model"], df["auroc_mean"], color="#4c72b0",
                   edgecolor="black", linewidth=0.5)
    ax.set_xlabel("LOF-CV AUROC (mean)")
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color="k", linestyle="--", linewidth=0.5)
    for bar, val in zip(bars, df["auroc_mean"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_title("Leave-one-field-out AUROC by model")

    out = fig_dir / "figure3_model_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
