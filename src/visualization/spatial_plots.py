"""Figure 6 — spatial smoothing α sensitivity."""
from __future__ import annotations

import argparse
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
    figs = ROOT / "outputs/figures"
    figs.mkdir(parents=True, exist_ok=True)

    path = res / "alpha_sensitivity.csv"
    if not path.exists():
        print("[warn] alpha_sensitivity.csv missing")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(df["alpha"], df["auroc_mean"], yerr=df["auroc_std"],
                marker="o", linewidth=2, capsize=4)
    ax.set_xlabel("spatial smoothing α")
    ax.set_ylabel("LOF-CV AUROC")
    ax.set_title("Spatial smoothing sensitivity")
    ax.grid(alpha=0.3)
    out = figs / "figure6_alpha_sensitivity.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
