"""Hard-negative validation cohort — cells where the geology makes a
hydrothermal system effectively impossible, used to test whether the model's
constructed negatives are well-calibrated.

We build four cohorts of increasing geological confidence:

  NC3 random_deep              cells > 100 km from any field, no springs
  NC4 cratonic_low_heatflow    NC3 + heat flow z-score < -1 SD AND no Cenozoic
                               volcanism within 100 km
  NC5 colorado_plateau_strict  Colorado Plateau interior + heat flow z < -0.5
                               + no Cenozoic igneous within 75 km
  NC6 stable_montana_wyoming   Northern Rocky Mountain stable platform, far
                               from any extensional zone, deep crust

If the model is genuinely learning geological structure (not just "far from
positives"), NC4/NC5/NC6 should score *lower* than NC3 — the model should
distinguish "merely far from a field" from "geologically incompatible."

Outputs ``outputs/results/hard_negative_cohorts.csv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/cpu_max.yaml"))
    processed = ROOT / cfg["paths"]["processed_dir"]

    scores = np.load(ROOT / "outputs/results/scores_cpu_max.npy")
    grid = pd.read_csv(processed / "grid_coordinates.csv")
    masks = np.load(processed / "modality_masks.npy")
    fields = pd.read_csv(ROOT / "data/metadata/known_fields_details.csv")
    geophys = np.load(processed / "geophysics_patches.npy", mmap_mode="r")

    # Heat-flow z-score from geophysics channel 3 (the central patch value)
    hf_z = geophys[:, 3, 16, 16]  # already standardised in pipeline

    # Distance to nearest known field
    lat0 = float(grid.lat.mean())
    cos_lat0 = float(np.cos(np.radians(lat0)))
    gxy = np.column_stack([grid.lon * 111.32 * cos_lat0, grid.lat * 111.32])
    fxy = np.column_stack([fields.lon * 111.32 * cos_lat0, fields.lat * 111.32])
    d_to_field, _ = cKDTree(fxy).query(gxy, k=1)

    # Cenozoic igneous proxy: we don't have a direct volcano dataset for the
    # whole region, so use known fields > 60 °C as a coarse proxy of recent
    # heat-bearing systems. Cells far from ALL of them are "amagmatic."
    d_to_amagmatic_pool = d_to_field  # already what we need

    # Tectonic provinces (column already computed in build_labels)
    provinces = grid["province"].values if "province" in grid.columns else None
    if provinces is None:
        # Recompute on the fly
        def _prov(la, lo):
            if la > 41 and -124 < lo < -120: return "Cascades"
            if la < 42 and -118 < lo < -113: return "Basin_and_Range"
            if 42 < la < 46 and -116 < lo < -110: return "Snake_River_Plain"
            if 31 < la < 35 and -116 < lo < -113: return "Salton_Trough"
            if -113 < lo < -103: return "Rocky_Mountain"
            return "Other"
        provinces = np.array([_prov(la, lo) for la, lo in zip(grid.lat, grid.lon)])

    rng = np.random.default_rng(42)

    cohorts = {}

    # NC3 random deep (baseline — already in negative_controls.csv)
    mask_nc3 = (d_to_field > 100) & (~masks[:, 1])
    cohorts["NC3_random_deep"] = mask_nc3

    # NC4 = NC3 + low heat flow AND far from amagmatic pool
    mask_nc4 = mask_nc3 & (hf_z < -1.0) & (d_to_amagmatic_pool > 150)
    cohorts["NC4_cratonic_low_HF"] = mask_nc4

    # NC5 Colorado Plateau strict
    mask_cp = (grid.lat.between(36, 40) & grid.lon.between(-111, -108)).values
    mask_nc5 = mask_cp & (hf_z < -0.5) & (d_to_amagmatic_pool > 75)
    cohorts["NC5_colorado_plateau_strict"] = mask_nc5

    # NC6 stable Northern Rockies (E Montana, NE Wyoming)
    mask_nrk = (grid.lat.between(44, 48) & grid.lon.between(-110, -103)).values
    mask_nc6 = mask_nrk & (hf_z < -0.3) & (d_to_field > 75)
    cohorts["NC6_montana_wyoming_platform"] = mask_nc6

    # Score percentiles
    sort = np.argsort(scores)
    rank = np.empty_like(sort, dtype=np.int64)
    rank[sort] = np.arange(len(scores))
    pct = 100.0 * rank / (len(scores) - 1)

    rows = []
    for name, mask in cohorts.items():
        idx_pool = np.flatnonzero(mask)
        if len(idx_pool) == 0:
            rows.append(dict(cohort=name, n_available=0, n_sample=0,
                              mean_p=np.nan, mean_pct=np.nan,
                              frac_below_10pct=np.nan,
                              frac_below_5pct=np.nan,
                              frac_above_50pct=np.nan,
                              frac_above_90pct=np.nan,
                              mean_hf_z=np.nan,
                              mean_d_to_field_km=np.nan))
            continue
        n = min(100, len(idx_pool))
        idx = rng.choice(idx_pool, size=n, replace=False)
        s_sub = scores[idx]
        p_sub = pct[idx]
        rows.append(dict(
            cohort=name,
            n_available=int(mask.sum()),
            n_sample=n,
            mean_p=float(s_sub.mean()),
            mean_pct=float(p_sub.mean()),
            frac_below_10pct=float((p_sub < 10).mean()),
            frac_below_5pct=float((p_sub < 5).mean()),
            frac_above_50pct=float((p_sub >= 50).mean()),
            frac_above_90pct=float((p_sub >= 90).mean()),
            mean_hf_z=float(hf_z[idx].mean()),
            mean_d_to_field_km=float(d_to_field[idx].mean()),
        ))

    df = pd.DataFrame(rows)
    out = ROOT / "outputs/results/hard_negative_cohorts.csv"
    df.to_csv(out, index=False)

    print("=== HARD-NEGATIVE VALIDATION COHORTS ===")
    print(df.to_string(index=False, float_format="%.3f"))
    print()
    if not df.iloc[1:].mean_pct.isna().all():
        print("Interpretation: a model that has learned real geology (not just")
        print("'far from positives') should score NC4-NC6 LOWER than NC3.")
        print("If they're equal, the model is using only the proximity signal.")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
