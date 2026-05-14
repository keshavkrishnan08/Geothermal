"""Reservoir-temperature sensitivity sweep for the 33 consensus discoveries.

A central reviewer concern: the headline 6.6 GW (P50) depends on a
single per-cluster T_central estimate. If T were systematically biased by
+- 20 deg C or +- 40 deg C, how does the cumulative MWe change?

Protocol
--------
For each cluster, re-run the Williams 2008 Monte-Carlo with T_central
shifted by {-40, -20, 0, +20, +40} deg C, holding area, thickness, RF
and eta priors fixed. Report per-cluster MWe envelope and the
cumulative ΣP50.

Outputs
-------
outputs/results/mwe_temperature_sensitivity.csv  -- per-cluster sweep
outputs/results/mwe_temperature_sensitivity_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.mwe_estimation_v2 import monte_carlo_mwe

ROOT = Path(__file__).resolve().parents[2]


def main():
    results = ROOT / "outputs/results"
    disc = pd.read_csv(results / "consensus_with_mwe_v2.csv")
    print(f"loaded {len(disc)} consensus discoveries")

    deltas = [-40, -20, 0, +20, +40]
    rng = np.random.default_rng(42)

    rows = []
    for _, c in disc.iterrows():
        row = {"discovery_id": int(c.discovery_id),
               "lat": c.lat, "lon": c.lon,
               "area_km2": c.area_km2,
               "T_central_original": c.T_res_central}
        for d in deltas:
            T_shift = float(np.clip(c.T_res_central + d, 50.0, 320.0))
            mc = monte_carlo_mwe(area_km2=c.area_km2, T_est_C=T_shift,
                                 n_samples=2000, rng=rng)
            row[f"T{d:+d}_P10"] = mc["mwe_p10"]
            row[f"T{d:+d}_P50"] = mc["mwe_p50"]
            row[f"T{d:+d}_P90"] = mc["mwe_p90"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(results / "mwe_temperature_sensitivity.csv", index=False)

    # Cumulative summary
    summary = {"deltas_C": deltas}
    for d in deltas:
        col_p50 = f"T{d:+d}_P50"
        col_p10 = f"T{d:+d}_P10"
        col_p90 = f"T{d:+d}_P90"
        summary[f"sum_P50_at_dT{d:+d}"] = float(df[col_p50].sum())
        summary[f"sum_P10_at_dT{d:+d}"] = float(df[col_p10].sum())
        summary[f"sum_P90_at_dT{d:+d}"] = float(df[col_p90].sum())

    print("\nCumulative MWe under T-sensitivity sweep:")
    print(f"  dT (deg C) | ΣP10 (MWe)  | ΣP50 (MWe)  | ΣP90 (MWe)")
    print(f"  ----------:|------------:|------------:|------------:")
    for d in deltas:
        print(f"  {d:+10d} | {summary[f'sum_P10_at_dT{d:+d}']:11,.0f} | "
              f"{summary[f'sum_P50_at_dT{d:+d}']:11,.0f} | "
              f"{summary[f'sum_P90_at_dT{d:+d}']:11,.0f}")

    p50_at_0 = summary["sum_P50_at_dT+0"]
    p50_at_minus40 = summary["sum_P50_at_dT-40"]
    p50_at_plus40 = summary["sum_P50_at_dT+40"]
    summary["sensitivity_pct_per_20C"] = float(
        (summary["sum_P50_at_dT+20"] - summary["sum_P50_at_dT-20"])
        / (2 * p50_at_0) * 100)
    print(f"\nResource is {summary['sensitivity_pct_per_20C']:.1f}% sensitive to "
          f"a +- 20 deg C T-shift; +- 40 deg C envelope is "
          f"{p50_at_minus40:,.0f} -- {p50_at_plus40:,.0f} MWe.")

    with open(results / "mwe_temperature_sensitivity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote outputs/results/mwe_temperature_sensitivity.csv")
    print(f"wrote outputs/results/mwe_temperature_sensitivity_summary.json")


if __name__ == "__main__":
    main()
