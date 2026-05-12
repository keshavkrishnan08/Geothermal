"""Format Table 2 — ablation study — as a publication-ready LaTeX snippet."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/results/table2_ablation.csv")
    parser.add_argument("--out", default="paper/tables/table2_ablation.tex")
    args = parser.parse_args()

    src = ROOT / args.csv
    if not src.exists():
        print(f"[warn] {src} missing")
        return
    df = pd.read_csv(src)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df["AUROC"] = df.apply(lambda r: f"{r['auroc_mean']:.4f} ± {r['auroc_std']:.4f}", axis=1)
    df["AUPRC"] = df.apply(lambda r: f"{r['auprc_mean']:.4f} ± {r['auprc_std']:.4f}", axis=1)
    tex = df[["variant", "description", "AUROC", "AUPRC"]].to_latex(
        index=False, escape=False,
        column_format="llcc",
        caption="Ablation study (mean ± std over LOF folds).",
        label="tab:ablation",
    )
    out.write_text(tex)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
