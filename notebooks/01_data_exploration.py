"""Kaggle/Jupyter notebook: data exploration (Phase 1).

Runs the data pipeline interactively and prints a coverage report:
    - per-modality availability
    - positive / negative class balance
    - province distribution
"""
# %%
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
processed = ROOT / "data" / "processed"

# %%
grid = pd.read_csv(processed / "grid_coordinates.csv")
print(f"Grid cells: {len(grid):,}")

masks = np.load(processed / "modality_masks.npy")
for i, name in enumerate(["geophysics", "geochemistry", "thermal", "geology"]):
    print(f"  {name:14s} coverage: {masks[:, i].mean():.3f}")

# %%
labels = np.load(processed / "labels.npy")
train_mask = np.load(processed / "train_mask.npy")
print(f"Positives (labeled): {int(labels[train_mask].sum())}")
print(f"Negatives (labeled): {int(train_mask.sum() - labels[train_mask].sum())}")
print(f"Class ratio: 1 : {(train_mask.sum() - labels[train_mask].sum()) / max(1, labels[train_mask].sum()):.1f}")

# %%
provinces = pd.read_csv(ROOT / "data" / "metadata" / "tectonic_provinces.csv")
print("\nPositive cells per province:")
prov_of_pos = provinces.loc[(labels == 1)]["province"]
print(prov_of_pos.value_counts())
