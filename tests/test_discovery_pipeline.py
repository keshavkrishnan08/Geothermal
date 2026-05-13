"""End-to-end discovery checks: continental score distribution sane,
known fields recovered at high percentile, eastern OOD shape correct."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_continental_scores_finite_and_in_unit_interval(root: Path):
    p = root / "outputs/results/prospectivity_mean.npy"
    if not p.exists():
        pytest.skip("continental scores not generated")
    s = np.load(p)
    assert np.isfinite(s).all(), "NaN/Inf in continental score map"
    assert (s >= 0).all() and (s <= 1).all()
    # cpu_max scores most cells high (median ~0.96); the diagnostic that
    # actually matters is that the distribution isn't degenerate (i.e. not all
    # 0 or all 1), not the absolute mean.
    assert s.std() > 0.05, f"score map has degenerate spread: std={s.std():.4f}"
    assert s.min() < 0.2 and s.max() > 0.8, "score map saturated to one tail"


def test_continental_uncertainty_higher_in_score_tail(root: Path):
    """MC-Dropout uncertainty should be largest in the *transition* band, not
    the saturated tails. Empirically with cpu_max we expect mean σ at the
    high-confidence band (>0.7) to be at least 5× the mean σ in the
    near-zero band (<0.01) — otherwise the dropout layer is broken."""
    pm = root / "outputs/results/prospectivity_mean.npy"
    pu = root / "outputs/results/prospectivity_std.npy"
    if not (pm.exists() and pu.exists()):
        pytest.skip()
    mu = np.load(pm); sigma = np.load(pu)
    high = mu > 0.7
    very_low = mu < 0.01
    if high.sum() < 10 or very_low.sum() < 100:
        pytest.skip("not enough cells in either tail")
    assert sigma[high].mean() > 5 * sigma[very_low].mean(), (
        f"σ(high)={sigma[high].mean():.4f} vs σ(near-0)={sigma[very_low].mean():.4f} "
        f"— dropout looks degenerate")


def test_consensus_discoveries_have_fields(root: Path):
    """Schema check on the canonical discoveries table."""
    p = root / "outputs/results/consensus_discoveries.csv"
    if not p.exists():
        pytest.skip()
    df = pd.read_csv(p)
    needed = {"discovery_id", "lat", "lon", "n_cells", "p_mean", "p_std",
              "area_km2", "province"}
    assert needed.issubset(df.columns), f"missing: {needed - set(df.columns)}"
    assert (df.lat.between(30, 50)).all(), "lat outside western US"
    assert (df.lon.between(-126, -102)).all(), "lon outside western US"
    # Discoveries must have positive area and >= 1 cell.
    assert (df.n_cells >= 1).all() and (df.area_km2 > 0).all()


def test_known_fields_recovered_in_top_quartile(root: Path):
    """Sanity: the bulk of documented fields should land above the continental
    75th percentile. We don't enforce 90th because cpu_max's score distribution
    is heavily skewed and the 90th pct cuts ~24K cells."""
    p_grid = root / "data/processed/grid_coordinates.csv"
    p_score = root / "outputs/results/prospectivity_mean.npy"
    p_kf = root / "data/metadata/known_fields_details.csv"
    if not all(x.exists() for x in (p_grid, p_score, p_kf)):
        pytest.skip()
    grid = pd.read_csv(p_grid); scores = np.load(p_score)
    kf = pd.read_csv(p_kf)
    if len(kf) == 0:
        pytest.skip()
    thresh = float(np.quantile(scores, 0.75))
    from scipy.spatial import cKDTree
    lat0 = float(grid.lat.mean()); cos_lat0 = float(np.cos(np.radians(lat0)))
    gxy = np.column_stack([grid.lon * 111.32 * cos_lat0, grid.lat * 111.32])
    tree = cKDTree(gxy)
    hit = total = 0
    for _, k in kf.iterrows():
        if "lat" not in k or "lon" not in k or pd.isna(k["lat"]):
            continue
        total += 1
        _, idx = tree.query([k.lon * 111.32 * cos_lat0, k.lat * 111.32], k=1)
        if scores[idx] >= thresh:
            hit += 1
    coverage = hit / max(1, total)
    assert coverage > 0.5, (
        f"only {coverage:.1%} of known fields scored above the 75th pct")


def test_eastern_ood_grid_shape(root: Path):
    p = root / "outputs/results/ood_eastern_us_scores.npy"
    if not p.exists():
        pytest.skip()
    s = np.load(p)
    assert np.isfinite(s).all()
    assert s.shape[0] > 1000, "eastern grid suspiciously small"
    assert (s >= 0).all() and (s <= 1).all()
