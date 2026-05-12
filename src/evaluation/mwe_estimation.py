"""USGS Williams (2008) volumetric MWe estimation per discovered site.

For each cluster center we estimate recoverable electrical power as

    Q_th = V * rho * c_p * (T_res - T_ref)         [thermal energy J]
    P_e  = Q_th * RF * CE / (lifetime * seconds_per_year)   [W_e]

with defaults (Williams 2008, Table 1):
    V          = A * thickness   (m^3)
    A          = cluster area    (m^2)
    thickness  = 250 m           (typical hydrothermal reservoir)
    rho        = 2700 kg/m^3
    c_p        = 1000 J/(kg K)
    T_ref      = 90 deg C        (rejection temperature for binary plants)
    RF         = 0.10            (recovery factor)
    CE         = 0.10            (conversion efficiency)
    lifetime   = 30 years

T_res is taken from nearby geothermometer features when available, otherwise
inferred from local heat flow via the 1-D conductive model at 2 km depth.

Outputs ``outputs/results/discoveries_with_mwe.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = dict(
    thickness_m=250.0,
    rho=2700.0,
    cp=1000.0,
    T_ref_C=90.0,
    recovery_factor=0.10,
    conversion_efficiency=0.10,
    lifetime_years=30.0,
)
SECONDS_PER_YEAR = 365.25 * 86400.0


def thermal_to_mwe(area_m2: float, T_res_C: float, **kwargs) -> float:
    p = {**DEFAULTS, **kwargs}
    V = area_m2 * p["thickness_m"]
    Q_th = V * p["rho"] * p["cp"] * max(0.0, T_res_C - p["T_ref_C"])
    P_e_W = Q_th * p["recovery_factor"] * p["conversion_efficiency"] \
            / (p["lifetime_years"] * SECONDS_PER_YEAR)
    return P_e_W / 1e6  # MW_e


def _estimate_T_res(cluster_lat: float, cluster_lon: float,
                    grid_xy: np.ndarray, geophys_idx: int,
                    geochem_features: np.ndarray, geophys_patches: np.ndarray,
                    cos_lat0: float) -> tuple[float, str]:
    """Return (T_res_C, source) using geothermometer if available; else
    derive from local heat flow + conductive model at 2 km."""
    px = cluster_lon * 111.32 * cos_lat0
    py = cluster_lat * 111.32
    tree = cKDTree(grid_xy)
    _, nearest_cell = tree.query([px, py], k=1)

    # Geothermometer features 2 (T_chalcedony) and 3 (T_NaK), z-scored — can't
    # invert without normalisation stats; we read them directly when available
    # by checking if the standardised value is materially > 0 (i.e. above the
    # training-set mean). Falls back to heat-flow-based estimate.
    chalc_z = float(geochem_features[nearest_cell, 2])
    nak_z = float(geochem_features[nearest_cell, 3])
    if chalc_z > 1.0 or nak_z > 1.0:
        # Above-average geothermometer signal → assume reservoir 150-220 deg C
        T_est = 150.0 + 35.0 * max(chalc_z, nak_z)
        return min(T_est, 280.0), "geothermometer"

    # Fall back: conductive temperature at 2 km from heat flow channel (3) of
    # geophysics — z-scored, so we use the centre-of-patch value as a proxy.
    hf_z = float(geophys_patches[nearest_cell, 3, 16, 16])
    # rough back-of-envelope: 80 mW/m^2 + 40 mW/m^2 per std → T(2km) = ~80 + 30 * hf_z
    T_est = 80.0 + 30.0 * hf_z
    return max(50.0, min(T_est, 220.0)), "heat_flow_proxy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--in_csv", default="outputs/results/table3_discoveries.csv")
    parser.add_argument("--out_csv", default="outputs/results/discoveries_with_mwe.csv")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))

    src = ROOT / args.in_csv
    if not src.exists():
        print(f"[warn] {src} missing — run discovery first")
        return
    discoveries = pd.read_csv(src)
    if len(discoveries) == 0:
        print("[warn] no discoveries to score")
        return

    processed = ROOT / cfg["paths"]["processed_dir"]
    grid_df = pd.read_csv(processed / "grid_coordinates.csv")
    geochem = np.load(processed / "geochemistry_features.npy", mmap_mode="r")
    geophys = np.load(processed / "geophysics_patches.npy", mmap_mode="r")

    lat0 = float(grid_df["lat"].mean())
    cos_lat0 = float(np.cos(np.radians(lat0)))
    grid_xy = np.column_stack([
        grid_df["lon"].values * 111.32 * cos_lat0,
        grid_df["lat"].values * 111.32,
    ])

    rows = []
    for _, d in discoveries.iterrows():
        T_res, src_label = _estimate_T_res(
            d["lat"], d["lon"], grid_xy, geophys_idx=0,
            geochem_features=geochem, geophys_patches=geophys,
            cos_lat0=cos_lat0,
        )
        area_m2 = float(d["area_km2"]) * 1e6
        mwe = thermal_to_mwe(area_m2, T_res)
        # Crude uncertainty band: ±25 deg C on T_res, ±50% on RF*CE
        mwe_low = thermal_to_mwe(area_m2, T_res - 25.0,
                                 recovery_factor=0.05, conversion_efficiency=0.05)
        mwe_high = thermal_to_mwe(area_m2, T_res + 25.0,
                                  recovery_factor=0.15, conversion_efficiency=0.15)
        rows.append({
            **d.to_dict(),
            "T_res_estimate_C": T_res,
            "T_res_source": src_label,
            "mwe_central": mwe,
            "mwe_low": mwe_low,
            "mwe_high": mwe_high,
        })

    out = pd.DataFrame(rows).sort_values("mwe_central", ascending=False)
    out_path = ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    total = float(out["mwe_central"].sum())
    total_low = float(out["mwe_low"].sum())
    total_high = float(out["mwe_high"].sum())
    print(out[["discovery_id", "lat", "lon", "T_res_estimate_C",
               "mwe_central", "mwe_low", "mwe_high",
               "province", "plausibility"]].to_string(index=False))
    print(f"\nCumulative recoverable resource: {total:.0f} MWe "
          f"[{total_low:.0f} - {total_high:.0f}] across {len(out)} candidates")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
