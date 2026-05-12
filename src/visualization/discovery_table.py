"""Format Table 3 — discovered sites — as LaTeX."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/results/table3_discoveries.csv")
    parser.add_argument("--out", default="paper/tables/table3_discoveries.tex")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    src = ROOT / args.csv
    if not src.exists():
        print(f"[warn] {src} missing")
        return
    df = pd.read_csv(src).head(args.top)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df["coord"] = df.apply(lambda r: f"{r['lat']:.3f}, {r['lon']:.3f}", axis=1)
    df["p"] = df.apply(lambda r: f"{r['p_mean']:.3f} ± {r['p_std']:.3f}", axis=1)
    tex = df[["discovery_id", "coord", "area_km2", "p", "province", "plausibility"]].to_latex(
        index=False, escape=False, column_format="lcccll",
        caption="Top discovered prospective sites identified by GeoProspectNet.",
        label="tab:discoveries",
    )
    out.write_text(tex)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
