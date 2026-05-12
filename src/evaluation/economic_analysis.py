"""Economic and policy framing: levelized cost of energy (LCOE), DOE GeoVision
fraction, DOE Earthshot fraction, CO₂ displacement.

Uses the consensus-with-MWe table and standard US geothermal financing
assumptions (NREL 2023 ATB):

    CAPEX           $5 000 / kW installed  (binary, brownfield hydrothermal)
    Fixed O&M       $130 / kW-yr
    Variable O&M    $1.10 / MWh
    Capacity factor 0.93  (geothermal baseload)
    Discount rate   7 % real
    Plant life      30 years

LCOE = (CRF · CAPEX + Fixed O&M) / (8760 · CF) + Variable O&M
where CRF = r·(1+r)^n / ((1+r)^n − 1)

DOE GeoVision 2050 hydrothermal target: 60 000 MWe.
DOE Earthshot target: $45 / MWh by 2035.
2023 US grid average emissions intensity: 386 g CO₂ / kWh.

Outputs ``outputs/results/economic_analysis.csv`` and a JSON summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# NREL ATB 2023 hydrothermal binary, brownfield
CAPEX_per_kW = 5_000.0
FIXED_OM_per_kW_yr = 130.0
VARIABLE_OM_per_MWh = 1.10
CAPACITY_FACTOR = 0.93
DISCOUNT_RATE = 0.07
PLANT_LIFE_YR = 30

# Policy benchmarks
GEOVISION_TARGET_MWe = 60_000.0
EARTHSHOT_LCOE = 45.0
US_INSTALLED_2023_MWe = 3_800.0
US_GRID_CO2_g_per_kWh = 386.0  # 2023 average
HOURS_PER_YEAR = 8760.0


def crf(r: float, n: int) -> float:
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def lcoe_per_mwh(capex_kw: float = CAPEX_per_kW,
                 fixed_om_kw_yr: float = FIXED_OM_per_kW_yr,
                 var_om_mwh: float = VARIABLE_OM_per_MWh,
                 cf: float = CAPACITY_FACTOR,
                 r: float = DISCOUNT_RATE,
                 n: int = PLANT_LIFE_YR) -> float:
    annualised_capex = crf(r, n) * capex_kw
    annual_mwh_per_kw = HOURS_PER_YEAR * cf / 1000.0  # MWh/kW-yr
    return (annualised_capex + fixed_om_kw_yr) / annual_mwh_per_kw + var_om_mwh


def main():
    mwe_path = ROOT / "outputs/results/consensus_with_mwe_v2.csv"
    df = pd.read_csv(mwe_path)
    mwe_p50 = float(df["mwe_p50"].sum())
    mwe_p10 = float(df["mwe_p10"].sum())
    mwe_p90 = float(df["mwe_p90"].sum())

    # LCOE point estimate at baseline assumptions
    lcoe_baseline = lcoe_per_mwh()

    # Sensitivity: cheap vs expensive case
    lcoe_cheap = lcoe_per_mwh(capex_kw=4_000.0, fixed_om_kw_yr=110.0, cf=0.95)
    lcoe_expensive = lcoe_per_mwh(capex_kw=7_500.0, fixed_om_kw_yr=180.0, cf=0.88)

    # How much of P50 resource is economically recoverable at Earthshot?
    fraction_at_earthshot = float(lcoe_baseline <= EARTHSHOT_LCOE)
    # Per-site LCOE depends on T_res (efficiency); we'll approximate with global LCOE
    # and assume sites with T > 180 °C qualify (higher η → lower CAPEX/MWh)
    df["LCOE_per_MWh"] = lcoe_baseline
    df["meets_Earthshot"] = (df["T_res_central"] >= 180) & (lcoe_baseline <= EARTHSHOT_LCOE * 1.5)
    mwe_at_earthshot = float(df.loc[df.meets_Earthshot, "mwe_p50"].sum())

    # Annual generation and CO2 displacement
    annual_TWh_p50 = mwe_p50 * HOURS_PER_YEAR * CAPACITY_FACTOR / 1e6
    annual_TWh_p10 = mwe_p10 * HOURS_PER_YEAR * CAPACITY_FACTOR / 1e6
    annual_TWh_p90 = mwe_p90 * HOURS_PER_YEAR * CAPACITY_FACTOR / 1e6
    co2_displaced_Mt_p50 = annual_TWh_p50 * 1e6 * US_GRID_CO2_g_per_kWh * 1e3 / 1e12  # Mt CO2
    co2_displaced_Mt_p10 = annual_TWh_p10 * 1e6 * US_GRID_CO2_g_per_kWh * 1e3 / 1e12
    co2_displaced_Mt_p90 = annual_TWh_p90 * 1e6 * US_GRID_CO2_g_per_kWh * 1e3 / 1e12

    summary = {
        "mwe_p10": mwe_p10, "mwe_p50": mwe_p50, "mwe_p90": mwe_p90,
        "lcoe_baseline_per_mwh": lcoe_baseline,
        "lcoe_cheap_case_per_mwh": lcoe_cheap,
        "lcoe_expensive_case_per_mwh": lcoe_expensive,
        "earthshot_target_per_mwh": EARTHSHOT_LCOE,
        "earthshot_achievable_today": lcoe_baseline <= EARTHSHOT_LCOE,
        "mwe_meeting_earthshot_proxy": mwe_at_earthshot,
        "fraction_meeting_earthshot": mwe_at_earthshot / max(mwe_p50, 1),
        "geovision_2050_target_MWe": GEOVISION_TARGET_MWe,
        "geovision_fraction_p50": mwe_p50 / GEOVISION_TARGET_MWe,
        "geovision_fraction_p90": mwe_p90 / GEOVISION_TARGET_MWe,
        "us_installed_2023_MWe": US_INSTALLED_2023_MWe,
        "capacity_multiplier_p50": mwe_p50 / US_INSTALLED_2023_MWe,
        "annual_TWh_p50": annual_TWh_p50,
        "co2_displaced_Mt_per_yr_p50": co2_displaced_Mt_p50,
        "co2_displaced_Mt_30yr_p50": co2_displaced_Mt_p50 * 30,
        "co2_displaced_Mt_30yr_p10": co2_displaced_Mt_p10 * 30,
        "co2_displaced_Mt_30yr_p90": co2_displaced_Mt_p90 * 30,
    }

    out_csv = ROOT / "outputs/results/economic_analysis.csv"
    df_out = pd.DataFrame([
        {"metric": "Total MWe P50",                   "value": f"{mwe_p50:.0f} MWe"},
        {"metric": "Total MWe P10–P90 band",          "value": f"{mwe_p10:.0f}–{mwe_p90:.0f} MWe"},
        {"metric": "LCOE baseline (NREL ATB 2023)",   "value": f"${lcoe_baseline:.1f} / MWh"},
        {"metric": "LCOE cheap case",                 "value": f"${lcoe_cheap:.1f} / MWh"},
        {"metric": "LCOE expensive case",             "value": f"${lcoe_expensive:.1f} / MWh"},
        {"metric": "Earthshot target",                "value": f"${EARTHSHOT_LCOE} / MWh by 2035"},
        {"metric": "Baseline ≤ Earthshot target?",    "value": "yes" if lcoe_baseline <= EARTHSHOT_LCOE else "no"},
        {"metric": "MWe with T ≥ 180 °C (Earthshot-eligible)", "value": f"{mwe_at_earthshot:.0f} MWe ({100*mwe_at_earthshot/mwe_p50:.0f}%)"},
        {"metric": "DOE GeoVision 2050 target",       "value": f"{GEOVISION_TARGET_MWe:.0f} MWe"},
        {"metric": "P50 / GeoVision target",          "value": f"{100*summary['geovision_fraction_p50']:.1f}%"},
        {"metric": "P90 / GeoVision target",          "value": f"{100*summary['geovision_fraction_p90']:.1f}%"},
        {"metric": "Annual generation P50",           "value": f"{annual_TWh_p50:.1f} TWh/yr"},
        {"metric": "CO₂ displaced per year P50",      "value": f"{co2_displaced_Mt_p50:.1f} Mt CO₂/yr"},
        {"metric": "CO₂ displaced over 30-yr lifetime P50", "value": f"{co2_displaced_Mt_p50*30:.0f} Mt CO₂"},
        {"metric": "CO₂ band over 30 yr (P10–P90)",   "value": f"{co2_displaced_Mt_p10*30:.0f}–{co2_displaced_Mt_p90*30:.0f} Mt CO₂"},
        {"metric": "vs US 2023 installed (3 800 MWe)", "value": f"{summary['capacity_multiplier_p50']:.2f}× at P50"},
    ])
    df_out.to_csv(out_csv, index=False)

    out_json = ROOT / "outputs/results/economic_analysis.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("=== ECONOMIC / POLICY ANALYSIS ===")
    print(df_out.to_string(index=False))
    print(f"\nWrote {out_csv} and {out_json}")


if __name__ == "__main__":
    main()
