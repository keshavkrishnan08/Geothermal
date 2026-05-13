"""Catch silent data-pipeline regressions: NaN labels, out-of-range coordinates,
mismatched array lengths, invalid mask values."""
from __future__ import annotations

import numpy as np


def test_grid_array_lengths_match(processed):
    labels = np.load(processed / "labels.npy")
    masks = np.load(processed / "modality_masks.npy")
    train_mask = np.load(processed / "train_mask.npy")
    geo = np.load(processed / "geophysics_patches.npy", mmap_mode="r")
    chem = np.load(processed / "geochemistry_features.npy", mmap_mode="r")
    geol = np.load(processed / "geology_features.npy", mmap_mode="r")
    n = len(labels)
    for arr, name in [(masks, "masks"), (train_mask, "train_mask"),
                      (geo, "geo"), (chem, "chem"), (geol, "geol")]:
        assert len(arr) == n, f"{name} length {len(arr)} != labels {n}"


def test_labels_are_binary(labels):
    uniq = set(np.unique(labels).tolist())
    assert uniq <= {0.0, 1.0}, f"labels contain non-binary values: {uniq}"


def test_no_nan_in_features(processed):
    geo = np.load(processed / "geophysics_patches.npy", mmap_mode="r")
    chem = np.load(processed / "geochemistry_features.npy", mmap_mode="r")
    geol = np.load(processed / "geology_features.npy", mmap_mode="r")
    # Sample 1000 random cells (full mmap scan is slow on CPU).
    rng = np.random.default_rng(0)
    n = len(geo)
    idx = rng.choice(n, size=1000, replace=False)
    assert np.isfinite(np.array(geo[idx])).all(), "NaN in geophysics patches"
    assert np.isfinite(np.array(chem[idx])).all(), "NaN in geochemistry features"
    assert np.isfinite(np.array(geol[idx])).all(), "NaN in geology features"


def test_masks_are_boolean(processed):
    masks = np.load(processed / "modality_masks.npy")
    assert masks.dtype in (np.bool_, np.uint8, np.int8, np.int32, np.int64), \
        f"unexpected mask dtype {masks.dtype}"
    uniq = set(np.unique(masks).tolist())
    assert uniq <= {0, 1, True, False}, f"non-boolean values in mask: {uniq}"


def test_grid_coords_in_west_us_box(grid_coords):
    assert (grid_coords.lat >= 30.0).all() and (grid_coords.lat <= 50.0).all()
    assert (grid_coords.lon >= -126.0).all() and (grid_coords.lon <= -102.0).all()


def test_train_mask_covers_known_positives(processed, labels):
    """Every positive cell must be in the labeled training universe — otherwise
    we'd silently throw away ground-truth fields."""
    train_mask = np.load(processed / "train_mask.npy").astype(bool)
    pos = labels.astype(bool)
    assert pos.sum() > 0
    assert (pos & ~train_mask).sum() == 0, (
        f"{(pos & ~train_mask).sum()} positives are outside the train_mask")


def test_neighbor_indices_in_bounds(processed):
    n = int(np.load(processed / "labels.npy", mmap_mode="r").shape[0])
    nb = np.load(processed / "neighbor_indices.npy")
    assert nb.shape[0] == n
    assert nb.min() >= 0 and nb.max() < n
