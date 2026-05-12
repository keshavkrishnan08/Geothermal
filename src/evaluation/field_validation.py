"""Field validation: cross-check consensus discoveries against public Nevada
geothermal permit records (nv_permits_extracted.csv).

A consensus discovery is "field-validated" if any Nevada exploration or
production permit lies within 25 km of its centroid and was filed AFTER 2008
(i.e. not in our positive training set). Hits are reported and saved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]


def main():
    discoveries = pd.read_csv(ROOT / "outputs/results/consensus_discoveries.csv")
    permits = pd.read_csv(ROOT / "data/raw/labels/nv_permits_extracted.csv")
    permits = permits.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # Project to local km
    lat0 = float(discoveries.lat.mean())
    cos_lat0 = float(np.cos(np.radians(lat0)))
    d_xy = np.column_stack([discoveries.lon * 111.32 * cos_lat0, discoveries.lat * 111.32])
    p_xy = np.column_stack([permits.lon * 111.32 * cos_lat0, permits.lat * 111.32])

    tree = cKDTree(p_xy)
    # Find permits within 25 km of each discovery
    dists, idxs = tree.query(d_xy, k=5)  # 5 nearest permits per discovery

    rows = []
    for di, disc in discoveries.iterrows():
        nearest = []
        for d, i in zip(dists[di], idxs[di]):
            if d <= 25.0:
                p = permits.iloc[i]
                nearest.append(f"{p.permit if pd.notna(p.permit) else 'permit?'} "
                              f"{p.well} ({p.operator}) "
                              f"{p.county} @ {d:.1f}km")
        rows.append({
            "discovery_id": int(disc.discovery_id),
            "lat": disc.lat, "lon": disc.lon,
            "province": disc.province,
            "p_mean": disc.p_mean,
            "n_nearby_permits_25km": int(sum(d <= 25 for d in dists[di])),
            "nearest_permit_km": float(dists[di][0]),
            "nearest_permits": "; ".join(nearest) if nearest else "—",
        })
    df = pd.DataFrame(rows).sort_values("n_nearby_permits_25km", ascending=False)
    out_csv = ROOT / "outputs/results/field_validation.csv"
    df.to_csv(out_csv, index=False)

    n_validated = int((df.n_nearby_permits_25km > 0).sum())
    print(f"=== FIELD VALIDATION via Nevada permit records ===")
    print(f"  consensus discoveries: {len(df)}")
    print(f"  with ≥1 permit within 25 km: {n_validated} ({n_validated/len(df):.0%})")
    print(f"  mean nearest-permit distance: {df.nearest_permit_km.mean():.1f} km")
    print()
    print("--- consensus discoveries with permit corroboration ---")
    hits = df[df.n_nearby_permits_25km > 0]
    if len(hits):
        print(hits[["discovery_id", "lat", "lon", "province",
                    "n_nearby_permits_25km", "nearest_permit_km",
                    "nearest_permits"]].to_string(index=False))
    print()
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
