"""Score post-2008 NV permit sites against the trained model.

For each held-out permit cell, compute:
  - the predicted favourability percentile (across all 235K continental cells)
  - whether it lands in top {1, 5, 10, 25}%
"""
import sys, json
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np, pandas as pd, torch
import yaml
from scipy.spatial import cKDTree
from src.data.dataset import GeoProspectDataset, make_loader
from src.models.geoprospectnet import GeoProspectNet

cfg = yaml.safe_load(open("configs/cpu_smoke.yaml"))
device = "cuda" if torch.cuda.is_available() else "cpu"

# Find latest random-split checkpoint
ckpts = list(Path("outputs/checkpoints").glob("random_seed*_best.pt"))
if not ckpts:
    print("No random-split checkpoint yet; using newest fold checkpoint as proxy")
    ckpts = sorted(Path("outputs/checkpoints").glob("*_best.pt"), key=lambda p: p.stat().st_mtime)
ckpt = ckpts[-1]
print(f"loading checkpoint: {ckpt}")
ck = torch.load(ckpt, map_location=device, weights_only=False)
model = GeoProspectNet(cfg).to(device)
model.load_state_dict(ck["state_dict"])
model.eval()

# Continental inference
grid = pd.read_csv("data/processed/grid_coordinates.csv")
ds = GeoProspectDataset(Path("data/processed"), indices=np.arange(len(grid)),
                        use_thermal=cfg["modalities"]["use_thermal"])
loader = make_loader(ds, batch_size=512, shuffle=False, num_workers=0)
print(f"scoring {len(grid):,} cells ...")
scores = []
with torch.no_grad():
    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        scores.append(torch.sigmoid(out["logits"]).cpu().numpy())
scores = np.concatenate(scores)
print(f"score range: {scores.min():.4f} to {scores.max():.4f}, mean={scores.mean():.4f}")

# Load hold-out and find nearest grid cell for each
holdout = pd.read_csv("data/raw/labels/nv_permits_holdout.csv")
novel = holdout[holdout.is_novel].reset_index(drop=True)
print(f"novel sites: {len(novel)}")

lat0 = float(grid.lat.mean()); cos_lat0 = float(np.cos(np.radians(lat0)))
gxy = np.column_stack([grid.lon * 111.32 * cos_lat0, grid.lat * 111.32])
tree = cKDTree(gxy)

results = []
for _, row in novel.iterrows():
    px = row.lon * 111.32 * cos_lat0
    py = row.lat * 111.32
    d, idx = tree.query([px, py], k=1)
    p = float(scores[idx])
    pct = float(100.0 * (scores < p).mean())
    results.append({
        "permit": row.get("permit", ""),
        "well": row.get("well", ""),
        "operator": str(row.get("operator", ""))[:30],
        "lat": row.lat, "lon": row.lon,
        "predicted_p": p,
        "percentile": pct,
        "in_top_1pct": pct >= 99.0,
        "in_top_5pct": pct >= 95.0,
        "in_top_10pct": pct >= 90.0,
        "in_top_25pct": pct >= 75.0,
    })
out = pd.DataFrame(results)
out.to_csv("outputs/results/temporal_holdout.csv", index=False)
print()
print(out.sort_values("percentile", ascending=False).to_string(index=False))
print()
print(f"Aggregate temporal-holdout capture rates:")
print(f"  in top  1%: {out.in_top_1pct.mean():.1%}  ({out.in_top_1pct.sum()}/{len(out)})")
print(f"  in top  5%: {out.in_top_5pct.mean():.1%}  ({out.in_top_5pct.sum()}/{len(out)})")
print(f"  in top 10%: {out.in_top_10pct.mean():.1%}  ({out.in_top_10pct.sum()}/{len(out)})")
print(f"  in top 25%: {out.in_top_25pct.mean():.1%}  ({out.in_top_25pct.sum()}/{len(out)})")
