"""Paired t-tests with Holm-Bonferroni correction + bootstrap CIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]


def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down correction."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, pvals[idx] * (m - rank))
        adj[idx] = min(running, 1.0)
    return adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/results")
    args = parser.parse_args()

    results_dir = ROOT / args.results_dir
    gpn = json.load(open(results_dir / "lof_cv_results.json"))
    bls = json.load(open(results_dir / "baseline_results.json"))

    # Index GeoProspectNet by fold_id
    gpn_by_fold = {r["fold_id"]: r["best"]["auroc"]
                   for r in gpn if not np.isnan(r["best"]["auroc"])}

    rows = []
    raw_p = []
    names = []
    for name, fold_metrics in bls.items():
        paired = []
        for m in fold_metrics:
            fid = m["fold_id"]
            if fid in gpn_by_fold and not np.isnan(m.get("auroc", float("nan"))):
                paired.append((gpn_by_fold[fid], m["auroc"]))
        if len(paired) < 5:
            continue
        gpn_v, base_v = zip(*paired)
        t, p = stats.ttest_rel(gpn_v, base_v)
        delta = np.array(gpn_v) - np.array(base_v)
        rows.append({"baseline": name, "n_folds": len(paired),
                     "delta_mean": float(delta.mean()),
                     "t_stat": float(t), "p_value": float(p)})
        raw_p.append(p)
        names.append(name)

    if raw_p:
        adj = holm_bonferroni(np.array(raw_p))
        for r, a in zip(rows, adj):
            r["p_holm"] = float(a)
            r["significant"] = bool(a < 0.05)
    df = pd.DataFrame(rows)
    out = ROOT / "outputs/results/statistical_tests.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
