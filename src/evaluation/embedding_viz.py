"""t-SNE visualization of fused embeddings (Figure 7)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

from src.data.dataset import GeoProspectDataset, make_loader
from src.models.geoprospectnet import GeoProspectNet
from src.training.utils import device_auto, load_config

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    Z, Y, C = [], [], []
    for batch in loader:
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        out = model(batch_dev)
        Z.append(out["z_fused"].cpu().numpy())
        Y.append(batch["label"].numpy())
        C.append(batch["cell_id"].numpy())
    return np.concatenate(Z), np.concatenate(Y), np.concatenate(C)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default="outputs/checkpoints/discovery_full.pt")
    parser.add_argument("--n_sample", type=int, default=10000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    processed = ROOT / cfg["paths"]["processed_dir"]

    ck = torch.load(ROOT / args.ckpt, map_location=device)
    model = GeoProspectNet(cfg).to(device)
    model.load_state_dict(ck["state_dict"])

    grid_df = pd.read_csv(processed / "grid_coordinates.csv")
    labels = np.load(processed / "labels.npy")
    train_mask = np.load(processed / "train_mask.npy")

    rng = np.random.default_rng(42)
    pos_idx = np.flatnonzero(train_mask & (labels == 1))
    neg_idx = rng.choice(np.flatnonzero(train_mask & (labels == 0)),
                         size=min(args.n_sample - len(pos_idx),
                                  int(train_mask.sum() - len(pos_idx))),
                         replace=False)
    idx = np.concatenate([pos_idx, neg_idx])

    ds = GeoProspectDataset(processed, indices=idx,
                            use_thermal=cfg["modalities"]["use_thermal"])
    loader = make_loader(ds, batch_size=256, shuffle=False, num_workers=0)
    Z, Y, C = extract_embeddings(model, loader, device)

    print(f"Running t-SNE on {len(Z)} embeddings ...")
    Z2 = TSNE(n_components=2, perplexity=30, init="pca",
              random_state=42, n_iter=1000).fit_transform(Z)

    out_dir = ROOT / "outputs/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(Z2[Y == 0, 0], Z2[Y == 0, 1], s=4, alpha=0.3, c="#888888", label="negative")
    ax.scatter(Z2[Y == 1, 0], Z2[Y == 1, 1], s=8, alpha=0.7, c="#d62728", label="positive")
    ax.set_title("Fused embeddings — by label")
    ax.legend()
    ax.set_xticks([]); ax.set_yticks([])

    # Province coloring
    def _prov(la, lo):
        if la > 41 and -124 < lo < -120: return "Cascades"
        if la < 42 and -118 < lo < -113: return "Basin_and_Range"
        if 42 < la < 46 and -116 < lo < -110: return "Snake_River_Plain"
        if 31 < la < 35 and -116 < lo < -113: return "Salton_Trough"
        if -113 < lo < -103: return "Rocky_Mountain"
        return "Other"

    cell_lookup = grid_df.set_index("cell_id")
    provinces = [
        _prov(cell_lookup.loc[c, "lat"], cell_lookup.loc[c, "lon"]) for c in C
    ]
    ax = axes[1]
    palette = {
        "Basin_and_Range": "#1f77b4",
        "Cascades": "#2ca02c",
        "Snake_River_Plain": "#ff7f0e",
        "Salton_Trough": "#9467bd",
        "Rocky_Mountain": "#8c564b",
        "Other": "#aaaaaa",
    }
    for prov, color in palette.items():
        mask = np.array(provinces) == prov
        if mask.sum():
            ax.scatter(Z2[mask, 0], Z2[mask, 1], s=4, alpha=0.5,
                       c=color, label=prov)
    ax.set_title("Fused embeddings — by tectonic province")
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    out = out_dir / "figure7_tsne.png"
    plt.savefig(out, dpi=200)
    plt.savefig(out_dir / "figure7_tsne.pdf")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
