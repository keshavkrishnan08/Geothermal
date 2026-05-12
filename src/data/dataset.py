"""PyTorch Dataset and DataLoader helpers for GeoProspectNet."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]


class GeoProspectDataset(Dataset):
    """Per-cell multi-modal dataset.

    Loads ``.npy`` arrays produced by the ``process_*`` scripts. Patches and
    features are memory-mapped (``mmap_mode='r'``) so very large arrays do not
    blow up RAM on Kaggle.
    """

    def __init__(self, processed_dir: Path, indices: Optional[np.ndarray] = None,
                 use_thermal: bool = True):
        self.processed_dir = Path(processed_dir)
        self.use_thermal = use_thermal

        self.geophysics = np.load(processed_dir / "geophysics_patches.npy", mmap_mode="r")
        self.geochemistry = np.load(processed_dir / "geochemistry_features.npy",
                                    mmap_mode="r")
        if use_thermal:
            therm_path = processed_dir / "thermal_patches.npy"
            if therm_path.exists():
                arr = np.load(therm_path, mmap_mode="r")
                # Sentinel: tiny 1x1x1x1 array means "thermal missing"
                if arr.shape[0] == 1 and arr.shape[-1] == 1:
                    self.thermal = None
                else:
                    self.thermal = arr
            else:
                self.thermal = None
        else:
            self.thermal = None
        self.geology = np.load(processed_dir / "geology_features.npy", mmap_mode="r")
        self.masks = np.load(processed_dir / "modality_masks.npy", mmap_mode="r")
        self.labels = np.load(processed_dir / "labels.npy", mmap_mode="r")

        if indices is None:
            indices = np.arange(len(self.labels))
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        cell = int(self.indices[i])
        m = self.masks[cell]
        item = {
            "cell_id": torch.tensor(cell, dtype=torch.long),
            "geophysics": torch.from_numpy(np.array(self.geophysics[cell], dtype=np.float32)),
            "geochemistry": torch.from_numpy(np.array(self.geochemistry[cell], dtype=np.float32)),
            "geology": torch.from_numpy(np.array(self.geology[cell], dtype=np.float32)),
            "geo_mask": torch.tensor(bool(m[0])),
            "chem_mask": torch.tensor(bool(m[1])),
            "therm_mask": torch.tensor(bool(m[2])),
            "struct_mask": torch.tensor(bool(m[3])),
            "label": torch.tensor(float(self.labels[cell])),
        }
        if self.use_thermal and self.thermal is not None:
            item["thermal"] = torch.from_numpy(np.array(self.thermal[cell], dtype=np.float32))
        else:
            item["thermal"] = torch.zeros(1, 64, 64)
        return item


def make_loader(dataset: GeoProspectDataset, batch_size: int = 256,
                shuffle: bool = True, num_workers: int = 2,
                oversample_positive: float = 1.0) -> DataLoader:
    """Build a DataLoader; positives are oversampled if requested."""
    if oversample_positive > 1.0 and shuffle:
        labels = np.array([float(dataset.labels[int(i)]) for i in dataset.indices])
        weights = np.where(labels > 0.5, oversample_positive, 1.0).astype(np.float64)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, drop_last=True, pin_memory=True)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=False, pin_memory=True)
