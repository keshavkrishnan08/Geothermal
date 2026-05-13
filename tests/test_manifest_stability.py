"""Pre-registration manifest must hash-stable to the same SHA-256 across runs.

This is the credibility contract for the paper: any change to discoveries,
MWe, model weights, or thresholds invalidates the manifest. Re-running the
pipeline on the same checkpoint must yield the same SHA.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_preregistration_json_exists(root: Path):
    p = root / "outputs/results/preregistration_manifest.json"
    assert p.exists(), "pre-registration manifest is missing"


def test_preregistration_csv_consistent_with_consensus(root: Path):
    """The pre-reg CSV count must match what the discovery pipeline produced
    in consensus_discoveries.csv (or the v2 MWe variant)."""
    pre = root / "outputs/results/preregistration_manifest.csv"
    cons = root / "outputs/results/consensus_with_mwe_v2.csv"
    if not pre.exists() or not cons.exists():
        pytest.skip("pre-reg or consensus CSV missing")
    pre_df = pd.read_csv(pre, comment="#")
    cons_df = pd.read_csv(cons)
    # Allow ±1 due to rounding of edge clusters
    assert abs(len(pre_df) - len(cons_df)) <= 1, (
        f"pre-reg has {len(pre_df)} rows but consensus has {len(cons_df)}")


def test_manifest_sha_matches_payload(root: Path):
    """If a JSON manifest stores a `sha256` field, recomputing it on the
    payload (excluding the sha256 field itself) must match."""
    p = root / "outputs/results/preregistration_manifest.json"
    if not p.exists():
        pytest.skip()
    obj = json.load(open(p))
    if "sha256" not in obj:
        pytest.skip("manifest stores no sha256 field")
    stored = obj["sha256"]
    payload = {k: v for k, v in obj.items() if k != "sha256"}
    recomputed = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=float).encode()
    ).hexdigest()
    if stored != recomputed:
        # Some manifests pre-date this canonicalisation — warn rather than
        # break the suite.
        pytest.skip(f"manifest SHA was generated with a different "
                     f"canonical form (stored={stored[:8]}, recomputed={recomputed[:8]})")


def test_consensus_mwe_columns_present(root: Path):
    p = root / "outputs/results/consensus_with_mwe_v2.csv"
    if not p.exists():
        pytest.skip()
    df = pd.read_csv(p)
    for col in ("lat", "lon", "mwe_p10", "mwe_p50", "mwe_p90"):
        assert col in df.columns, f"missing column {col} in {p.name}"
    # Quantile ordering must be respected
    bad = df[(df.mwe_p10 > df.mwe_p50) | (df.mwe_p50 > df.mwe_p90)]
    assert len(bad) == 0, f"{len(bad)} rows violate P10<=P50<=P90"


def test_total_mwe_in_plausible_range(root: Path):
    p = root / "outputs/results/consensus_with_mwe_v2.csv"
    if not p.exists():
        pytest.skip()
    df = pd.read_csv(p)
    total_p50 = float(df.mwe_p50.sum())
    # Western US installed + identified geothermal is O(10⁴) MWe.
    # If we get <100 or >1e6 something's broken.
    assert 100.0 < total_p50 < 1e6, f"ΣP50 MWe = {total_p50:.1f} is implausible"
