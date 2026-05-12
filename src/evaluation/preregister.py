"""Freeze the 33 consensus discoveries as a pre-registration manifest with a
SHA-256 hash, ISO timestamp, and locked coordinates. Future drilling outcomes
can be compared against this immutable record.

Outputs:
    outputs/results/preregistration_manifest.csv
    outputs/results/preregistration_manifest.json   (with provenance hash)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    src = ROOT / "outputs/results/consensus_with_mwe_v2.csv"
    df = pd.read_csv(src)

    columns = ["discovery_id", "lat", "lon", "area_km2", "T_res_central",
                "mwe_p10", "mwe_p50", "mwe_p90", "province", "plausibility"]
    manifest = df[columns].sort_values("discovery_id").reset_index(drop=True)

    # Stable bytes for hashing (sorted columns, sorted rows)
    payload = manifest.to_csv(index=False).encode("utf-8")
    sha256 = hashlib.sha256(payload).hexdigest()

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header = (
        f"# GeoProspectNet — pre-registered candidate discoveries\n"
        f"# frozen: {ts}\n"
        f"# sha256: {sha256}\n"
        f"# source: outputs/results/consensus_with_mwe_v2.csv\n"
        f"# protocol: each row is a candidate hydrothermal site. Drilling outcomes\n"
        f"#           reported after this date constitute independent validation of\n"
        f"#           the model's predictions. No post-hoc edits are permitted.\n"
    )

    out_csv = ROOT / "outputs/results/preregistration_manifest.csv"
    out_csv.write_text(header + manifest.to_csv(index=False))

    summary = {
        "sha256": sha256,
        "iso_timestamp": ts,
        "n_sites": int(len(manifest)),
        "cumulative_MWe_p50": float(manifest.mwe_p50.sum()),
        "cumulative_MWe_p10": float(manifest.mwe_p10.sum()),
        "cumulative_MWe_p90": float(manifest.mwe_p90.sum()),
        "plausible_count": int((manifest.plausibility == "plausible").sum()),
        "provinces": manifest.groupby("province").size().to_dict(),
        "drilling_outcome_protocol": (
            "A discovery is 'verified' if a producing geothermal well or "
            "successful exploration result is reported within 25 km of the "
            "manifest centroid after the frozen timestamp. A discovery is "
            "'falsified' if a dry hole or formally negative survey is "
            "reported within 5 km of the manifest centroid after the frozen "
            "timestamp."
        ),
    }

    out_json = ROOT / "outputs/results/preregistration_manifest.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"frozen: {ts}")
    print(f"sha256: {sha256}")
    print(f"n_sites: {len(manifest)}")
    print(f"cumulative P50 MWe: {summary['cumulative_MWe_p50']:.0f}")
    print(f"province breakdown: {summary['provinces']}")
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_json}")
    print()
    print("Top 5 sites:")
    print(manifest.head(5).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
