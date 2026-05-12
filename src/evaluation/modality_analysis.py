"""Permutation importance + attention-weight analysis.

Permutation importance: shuffle each modality's features across all locations,
re-score on a held-out split, measure ΔAUROC. Repeat ``n_repeats`` times and
report mean ± std.

Attention analysis: collect per-sample attention weights from the fusion
layer, group by tectonic province, and report mean weights per province.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

from src.data.dataset import GeoProspectDataset, make_loader
from src.models.geoprospectnet import MODALITY_ORDER, GeoProspectNet
from src.training.utils import device_auto, load_config

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def _score(model, loader, device):
    model.eval()
    s, y = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        s.append(torch.sigmoid(out["logits"]).cpu().numpy())
        y.append(batch["label"].cpu().numpy())
    return np.concatenate(y), np.concatenate(s)


def _permute_modality(ds: GeoProspectDataset, modality: str, rng) -> GeoProspectDataset:
    """Return a shallow copy of `ds` with one modality's tensor shuffled."""
    new = GeoProspectDataset(ds.processed_dir, indices=ds.indices, use_thermal=ds.use_thermal)
    perm = rng.permutation(len(ds.labels))
    if modality == "geophysics":
        new.geophysics = ds.geophysics[perm]  # mmap will materialize a copy
    elif modality == "geochemistry":
        new.geochemistry = ds.geochemistry[perm]
    elif modality == "thermal":
        if ds.thermal is not None:
            new.thermal = ds.thermal[perm]
    elif modality == "geology":
        new.geology = ds.geology[perm]
    return new


@torch.no_grad()
def attention_by_province(model, loader, grid_df, device):
    """Compute the mean attention weights per tectonic province."""
    model.eval()
    weights_per_cell = []
    cells_seen = []
    for batch in loader:
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        out = model(batch_dev)
        weights_per_cell.append(out["attention_weights"].cpu().numpy())
        cells_seen.append(batch["cell_id"].numpy())
    weights_per_cell = np.concatenate(weights_per_cell, axis=0)
    cells_seen = np.concatenate(cells_seen, axis=0)
    df = pd.DataFrame(weights_per_cell, columns=list(MODALITY_ORDER[: weights_per_cell.shape[1]]))
    df["cell_id"] = cells_seen
    df = df.merge(grid_df[["cell_id", "lat", "lon"]], on="cell_id", how="left")
    # Province bin (mirror of build_labels._province_bin)
    def _prov(la, lo):
        if la > 41 and -124 < lo < -120: return "Cascades"
        if la < 42 and -118 < lo < -113: return "Basin_and_Range"
        if 42 < la < 46 and -116 < lo < -110: return "Snake_River_Plain"
        if 31 < la < 35 and -116 < lo < -113: return "Salton_Trough"
        if -113 < lo < -103: return "Rocky_Mountain"
        return "Other"
    df["province"] = [_prov(la, lo) for la, lo in zip(df.lat, df.lon)]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default="outputs/checkpoints/discovery_full.pt")
    parser.add_argument("--n_repeats", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    processed = ROOT / cfg["paths"]["processed_dir"]

    ck = torch.load(ROOT / args.ckpt, map_location=device)
    model = GeoProspectNet(cfg).to(device)
    model.load_state_dict(ck["state_dict"])

    splits = json.load(open(processed / "splits" / "train_val_test.json"))
    val_idx = np.array(splits["val_idx"])
    val_ds = GeoProspectDataset(processed, indices=val_idx,
                                use_thermal=cfg["modalities"]["use_thermal"])
    val_loader = make_loader(val_ds, batch_size=cfg["training"]["batch_size"],
                             shuffle=False, num_workers=0)

    # Baseline AUROC
    y, s = _score(model, val_loader, device)
    base_auc = float(roc_auc_score(y, s))
    print(f"Baseline AUROC: {base_auc:.4f}")

    rng = np.random.default_rng(42)
    drops = []
    modalities = ["geophysics", "geochemistry", "geology"]
    if cfg["modalities"]["use_thermal"]:
        modalities.append("thermal")
    for mod in modalities:
        run_drops = []
        for _ in range(args.n_repeats):
            ds2 = _permute_modality(val_ds, mod, rng)
            loader2 = make_loader(ds2, batch_size=cfg["training"]["batch_size"],
                                  shuffle=False, num_workers=0)
            _, s2 = _score(model, loader2, device)
            run_drops.append(base_auc - float(roc_auc_score(y, s2)))
        drops.append({
            "modality": mod,
            "delta_auc_mean": float(np.mean(run_drops)),
            "delta_auc_std": float(np.std(run_drops)),
        })
        print(f"  Δ{mod:14s} = {drops[-1]['delta_auc_mean']:.4f} "
              f"± {drops[-1]['delta_auc_std']:.4f}")

    pd.DataFrame(drops).to_csv(ROOT / "outputs/results/permutation_importance.csv",
                               index=False)

    # ---- Attention by province on a representative subset ----
    grid_df = pd.read_csv(processed / "grid_coordinates.csv")
    sample_idx = rng.choice(len(grid_df), size=min(20000, len(grid_df)), replace=False)
    sample_ds = GeoProspectDataset(processed, indices=sample_idx,
                                   use_thermal=cfg["modalities"]["use_thermal"])
    sample_loader = make_loader(sample_ds, batch_size=cfg["training"]["batch_size"],
                                shuffle=False, num_workers=0)
    att_df = attention_by_province(model, sample_loader, grid_df, device)
    grouped = att_df.groupby("province")[list(MODALITY_ORDER[: att_df.shape[1] - 4])].mean()
    grouped.to_csv(ROOT / "outputs/results/attention_by_province.csv")
    print("\nAttention by province:")
    print(grouped.to_string())


if __name__ == "__main__":
    main()
