"""USGS-grade volumetric MWe estimation with temperature-dependent conversion
efficiency and proper Monte-Carlo uncertainty propagation.

Improvements over `mwe_estimation.py` (the screening version):

1.  **Temperature-dependent conversion efficiency** instead of a flat 10 %.
    We use the modern utilisation-efficiency form (Bertani 2012, Zarrouk &
    Moon 2014) calibrated to operating US geothermal plants:

        η(T) = 0.45 · η_Carnot(T, T_ref)
             = 0.45 · (T_res - T_ref) / (T_res + 273.15)

    capped at 0.18 (the empirical ceiling for triple-flash plants on 280 °C
    systems). At T = 90 °C this gives ≈ 0, at 180 °C ≈ 0.09, at 250 °C ≈
    0.13, at 300 °C ≈ 0.16 — matching the band Williams 2008 Table 2.

2.  **Plant-type-aware recovery factor** instead of a flat 0.10:
        T < 150 °C  → RF ~ Lognormal(μ=0.08, σ=0.30)  (binary)
        150 ≤ T < 220 → RF ~ Lognormal(μ=0.12, σ=0.30)  (flash)
        T ≥ 220 °C → RF ~ Lognormal(μ=0.17, σ=0.30)  (flash / vapour-dominated)

3.  **Reservoir-thickness distribution** instead of a flat 250 m:
        h ~ Lognormal(μ=ln 250, σ=0.55)  (range ~80 m to 750 m at 1σ,
        consistent with the Williams 2008 dataset of US producing systems).

4.  **Reservoir-temperature uncertainty**:
        T ~ Normal(T_est, σ=20 °C), truncated to [50, 320] °C.

5.  **Monte Carlo**: 1000 samples per discovery. Report the central P50 plus
    P10 / P90 bands (proper uncertainty quantiles, not multiplicative bands).

Outputs ``outputs/results/consensus_with_mwe_v2.csv`` with P10/P50/P90 and
all the input distributions per discovery.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]

SECONDS_PER_YEAR = 365.25 * 86400.0
RHO = 2700.0
CP = 1000.0
T_REF_C = 90.0
LIFETIME_YEARS = 30.0


def conversion_efficiency(T_res_C: np.ndarray, T_ref_C: float = T_REF_C) -> np.ndarray:
    """Temperature-dependent conversion efficiency.

    Form: η = 0.45 · η_Carnot(T_hot, T_cold), capped at 0.18.
    η_Carnot = (T_hot - T_cold) / (T_hot + 273.15), T in °C.
    """
    eta_carnot = np.clip(T_res_C - T_ref_C, 0.0, None) / (T_res_C + 273.15)
    return np.clip(0.45 * eta_carnot, 0.0, 0.18)


def recovery_factor(T_res_C: np.ndarray, rng: np.random.Generator,
                    n: int) -> np.ndarray:
    """Plant-type-aware recovery factor sampled per draw.

    Binary (T < 150)        → log-normal centred 0.08
    Flash (150 ≤ T < 220)   → log-normal centred 0.12
    Flash / vapour (T ≥ 220)→ log-normal centred 0.17
    """
    # Broadcast T_res to per-draw if scalar
    T = np.broadcast_to(T_res_C, (n,))
    centres = np.where(T < 150.0, 0.08,
                       np.where(T < 220.0, 0.12, 0.17))
    sigmas = np.full(n, 0.30)
    mus = np.log(centres)
    return rng.lognormal(mean=mus, sigma=sigmas)


def thermal_to_mwe_array(area_m2: float, T_res_C: np.ndarray,
                         thickness_m: np.ndarray, RF: np.ndarray,
                         eta: np.ndarray) -> np.ndarray:
    V = area_m2 * thickness_m
    Q_th = V * RHO * CP * np.clip(T_res_C - T_REF_C, 0.0, None)
    P_e_W = Q_th * RF * eta / (LIFETIME_YEARS * SECONDS_PER_YEAR)
    return P_e_W / 1e6


def _estimate_T_res(cluster_lat: float, cluster_lon: float,
                    grid_xy: np.ndarray,
                    geochem_features: np.ndarray, geophys_patches: np.ndarray,
                    cos_lat0: float) -> tuple[float, str]:
    """Estimate central reservoir temperature using geothermometers if
    geochemistry coverage exists; otherwise fall back to heat-flow proxy at 2 km."""
    px = cluster_lon * 111.32 * cos_lat0
    py = cluster_lat * 111.32
    _, nearest_cell = cKDTree(grid_xy).query([px, py], k=1)

    chalc_z = float(geochem_features[nearest_cell, 2])
    nak_z = float(geochem_features[nearest_cell, 3])
    if chalc_z > 1.0 or nak_z > 1.0:
        T_est = 150.0 + 35.0 * max(chalc_z, nak_z)
        return min(T_est, 280.0), "geothermometer"

    hf_z = float(geophys_patches[nearest_cell, 3, 16, 16])
    T_est = 80.0 + 30.0 * hf_z
    return max(50.0, min(T_est, 220.0)), "heat_flow_proxy"


def monte_carlo_mwe(area_km2: float, T_est_C: float, n_samples: int,
                    rng: np.random.Generator) -> dict:
    area_m2 = area_km2 * 1e6
    T_draws = np.clip(rng.normal(loc=T_est_C, scale=20.0, size=n_samples),
                      50.0, 320.0)
    thickness_draws = rng.lognormal(mean=np.log(250.0), sigma=0.55,
                                     size=n_samples)
    thickness_draws = np.clip(thickness_draws, 60.0, 1000.0)
    RF_draws = recovery_factor(T_draws, rng, n_samples)
    eta_draws = conversion_efficiency(T_draws)

    mwe = thermal_to_mwe_array(area_m2, T_draws, thickness_draws, RF_draws,
                               eta_draws)
    return {
        "mwe_p10": float(np.percentile(mwe, 10)),
        "mwe_p50": float(np.percentile(mwe, 50)),
        "mwe_p90": float(np.percentile(mwe, 90)),
        "mwe_mean": float(mwe.mean()),
        "T_res_central": T_est_C,
        "T_res_mean_draws": float(T_draws.mean()),
        "thickness_mean_m": float(thickness_draws.mean()),
        "RF_mean": float(RF_draws.mean()),
        "eta_mean": float(eta_draws.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_max.yaml")
    parser.add_argument("--in_csv", default="outputs/results/consensus_discoveries.csv")
    parser.add_argument("--out_csv", default="outputs/results/consensus_with_mwe_v2.csv")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    discoveries = pd.read_csv(ROOT / args.in_csv)
    print(f"Estimating MWe for {len(discoveries)} discoveries "
          f"with {args.n_samples} Monte-Carlo draws each")

    processed = ROOT / cfg["paths"]["processed_dir"]
    grid_df = pd.read_csv(processed / "grid_coordinates.csv")
    geochem = np.load(processed / "geochemistry_features.npy", mmap_mode="r")
    geophys = np.load(processed / "geophysics_patches.npy", mmap_mode="r")

    lat0 = float(grid_df.lat.mean())
    cos_lat0 = float(np.cos(np.radians(lat0)))
    grid_xy = np.column_stack([
        grid_df.lon.values * 111.32 * cos_lat0,
        grid_df.lat.values * 111.32,
    ])

    rng = np.random.default_rng(args.seed)
    rows = []
    for _, d in discoveries.iterrows():
        T_est, src = _estimate_T_res(d.lat, d.lon, grid_xy, geochem, geophys, cos_lat0)
        mc = monte_carlo_mwe(float(d.area_km2), T_est, args.n_samples, rng)
        rows.append({
            **d.to_dict(),
            "T_res_source": src,
            **mc,
        })

    out = pd.DataFrame(rows).sort_values("mwe_p50", ascending=False)
    out_path = ROOT / args.out_csv
    out.to_csv(out_path, index=False)

    print(out[["discovery_id", "lat", "lon", "area_km2", "T_res_central",
               "mwe_p10", "mwe_p50", "mwe_p90", "province"]]
          .to_string(index=False, float_format="%.1f"))

    p10_sum = float(out.mwe_p10.sum())
    p50_sum = float(out.mwe_p50.sum())
    p90_sum = float(out.mwe_p90.sum())
    mean_sum = float(out.mwe_mean.sum())
    print(f"\nCumulative recoverable resource (Monte-Carlo over {args.n_samples} draws/site):")
    print(f"  P10     {p10_sum:7.0f} MWe   (conservative)")
    print(f"  P50     {p50_sum:7.0f} MWe   (central estimate — Nature Energy headline)")
    print(f"  P90     {p90_sum:7.0f} MWe   (optimistic)")
    print(f"  mean    {mean_sum:7.0f} MWe")
    print(f"\n  US installed (2023): 3 800 MWe → P50 / installed = {p50_sum/3800:.2f}×")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
