"""Figure 5 — Permutation importance + attention weights by tectonic province."""
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

    # ---- (a) Permutation importance ---------------------------------
    perm_path = res / "permutation_importance.csv"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if perm_path.exists():
        perm = pd.read_csv(perm_path).sort_values("delta_auc_mean", ascending=True)
        ax1.barh(perm["modality"], perm["delta_auc_mean"],
                 xerr=perm["delta_auc_std"], color="#dd8452", edgecolor="black")
        ax1.set_xlabel("ΔAUROC (drop when modality shuffled)")
        ax1.set_title("(a) Permutation importance")
        ax1.axvline(0, color="k", linewidth=0.5)

    # ---- (b) Attention by province ---------------------------------
    att_path = res / "attention_by_province.csv"
    if att_path.exists():
        att = pd.read_csv(att_path, index_col=0)
        att.plot(kind="bar", stacked=True, ax=ax2,
                 colormap="tab10", edgecolor="black", linewidth=0.3)
        ax2.set_ylabel("Mean attention weight")
        ax2.set_title("(b) Attention weights by tectonic province")
        ax2.legend(loc="upper right", fontsize=8)
        plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    out = figs / "figure5_modality_importance.png"
    plt.savefig(out, dpi=200)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
