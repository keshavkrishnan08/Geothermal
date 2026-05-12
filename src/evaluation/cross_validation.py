"""Orchestrate the full leave-one-field-out cross-validation across models.

Reads ``outputs/results/lof_cv_results.json`` and ``baseline_results.json``,
aggregates per-fold metrics, and writes a Table-1 CSV ready for the paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _agg(rows, key="auroc"):
    vals = [r["best"][key] if "best" in r else r[key] for r in rows
            if not np.isnan((r["best"][key] if "best" in r else r[key]))]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/results")
    parser.add_argument("--out_csv", default="outputs/results/table1_main_results.csv")
    args = parser.parse_args()

    results_dir = ROOT / args.results_dir
    rows = []

    REPORTED = (
        "auroc", "auprc", "f1",
        "precision_at_k", "recall_at_k",
        "capture_top_1pct", "capture_top_5pct",
        "capture_top_10pct", "capture_top_25pct",
    )

    # ---- GeoProspectNet ---------------------------------------------------
    gpn_path = results_dir / "lof_cv_results.json"
    if gpn_path.exists():
        gpn = json.load(open(gpn_path))
        row = {"model": "GeoProspectNet"}
        for k in REPORTED:
            mu, sd = _agg(gpn, key=k)
            row[k] = f"{mu:.4f} ± {sd:.4f}" if not np.isnan(mu) else "—"
        rows.append(row)

    # ---- Baselines -------------------------------------------------------
    bl_path = results_dir / "baseline_results.json"
    if bl_path.exists():
        bls = json.load(open(bl_path))
        for name, fold_metrics in bls.items():
            row = {"model": name}
            for k in REPORTED:
                vals = [m.get(k, float("nan")) for m in fold_metrics
                        if not np.isnan(m.get(k, float("nan")))]
                row[k] = "—" if not vals else f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
