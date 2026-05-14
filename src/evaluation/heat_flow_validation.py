"""Leave-one-out cross-validation of the IDW heat-flow interpolation.

A reviewer will rightly ask: ``Your heat-flow input map is interpolated
from SMU point stations via inverse-distance kNN. How accurate is the
interpolation?'' This script answers it.

Protocol
--------
1. Load the cleaned SMU heat-flow point dataset (lat, lon, SiteHeatFlow).
2. For each station inside the western US study box: predict its
   SiteHeatFlow by IDW-kNN (k=5, 200 km cutoff) using *all other stations*.
3. Report RMSE, MAE, R^2, and the residual distribution (% within ±10,
   ±20, ±30 mW/m^2 -- the bands that actually matter for binary-cycle vs.
   flash-plant classification).

The same kNN parameters are used in ``ood_eastern_us.sample_heat_flow``,
so this is a direct accuracy measurement on the feature the headline
model is fed.

Output
------
outputs/results/heat_flow_loo_validation.csv  -- per-station residuals
outputs/results/heat_flow_loo_summary.json    -- summary stats
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]


def main():
    raw = pd.read_csv(ROOT / "data/raw/geophysics/smu_heatflow/heat_flow_combined.csv",
                       usecols=["LatDegreeWGS84", "LongDegreeWGS84", "SiteHeatFlow"])
    raw = raw.rename(columns={"LatDegreeWGS84": "lat", "LongDegreeWGS84": "lon",
                                "SiteHeatFlow": "hf"})
    raw["hf"] = pd.to_numeric(raw["hf"], errors="coerce")
    # Filter: physically plausible band and inside the western US box
    valid = (raw["hf"] > 0) & (raw["hf"] < 500) \
            & raw["lat"].between(31.0, 49.0) \
            & raw["lon"].between(-125.0, -103.0) \
            & raw["lat"].notna() & raw["lon"].notna()
    df = raw[valid].reset_index(drop=True)
    print(f"valid HF stations in western US: {len(df):,}")

    lat0 = float(df.lat.mean())
    cos_lat0 = float(np.cos(np.radians(lat0)))
    xy = np.column_stack([df.lon.values * 111.32 * cos_lat0,
                           df.lat.values * 111.32])
    hf = df.hf.values.astype(np.float64)

    # Use a kNN with k+1; the first neighbour will be self, drop it.
    tree = cKDTree(xy)
    k = 5
    d, idx = tree.query(xy, k=k + 1)
    d = d[:, 1:]  # drop self
    idx = idx[:, 1:]
    # 200 km cutoff
    far = d[:, 0] > 200.0
    weights = 1.0 / (d + 1.0)
    pred = (hf[idx] * weights).sum(axis=1) / weights.sum(axis=1)
    pred[far] = np.nan

    valid_pred = ~np.isnan(pred)
    resid = hf[valid_pred] - pred[valid_pred]

    rmse = float(np.sqrt((resid ** 2).mean()))
    mae = float(np.abs(resid).mean())
    bias = float(resid.mean())
    pct_within = lambda b: float(100.0 * (np.abs(resid) <= b).mean())
    r2 = float(1.0 - (resid ** 2).sum()
                  / ((hf[valid_pred] - hf[valid_pred].mean()) ** 2).sum())

    print(f"\nLOO validation (n = {int(valid_pred.sum()):,}):")
    print(f"  RMSE:        {rmse:.1f}  mW/m^2")
    print(f"  MAE:         {mae:.1f}  mW/m^2")
    print(f"  Bias:        {bias:+.2f} mW/m^2")
    print(f"  R^2:         {r2:.3f}")
    print(f"  Within +- 10: {pct_within(10):.1f}%")
    print(f"  Within +- 20: {pct_within(20):.1f}%")
    print(f"  Within +- 30: {pct_within(30):.1f}%")
    print(f"  Within +- 50: {pct_within(50):.1f}%")

    out = pd.DataFrame({
        "lat": df.lat.values[valid_pred],
        "lon": df.lon.values[valid_pred],
        "hf_obs": hf[valid_pred],
        "hf_pred": pred[valid_pred],
        "residual": resid,
    })
    out.to_csv(ROOT / "outputs/results/heat_flow_loo_validation.csv", index=False)

    summary = {
        "n_stations": int(valid_pred.sum()),
        "rmse_mWm2": rmse,
        "mae_mWm2": mae,
        "bias_mWm2": bias,
        "r2": r2,
        "pct_within_10_mWm2": pct_within(10),
        "pct_within_20_mWm2": pct_within(20),
        "pct_within_30_mWm2": pct_within(30),
        "pct_within_50_mWm2": pct_within(50),
        "knn_k": k,
        "cutoff_km": 200.0,
    }
    with open(ROOT / "outputs/results/heat_flow_loo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote outputs/results/heat_flow_loo_validation.csv")
    print(f"Wrote outputs/results/heat_flow_loo_summary.json")


if __name__ == "__main__":
    main()
