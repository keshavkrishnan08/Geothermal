"""True single-modality ablation.

For each modality m in {geophysics, geochemistry, geology}, train a fresh
GeoProspectNet where the other two modalities are replaced by their learned
defaults via the model's ``modalities.drop`` mechanism. Contrastive loss is
disabled because there are no positive cross-modal pairs in the
single-modality regime. We use the same random 70/15/15 split as the
headline cpu_max model so the ΔAUROC is apples-to-apples.

Usage
-----
    python -m src.training.single_modality_ablation --config configs/cpu_max.yaml
        [--epochs 40]            # cap epochs for overnight CPU
        [--patience 8]           # early-stop patience
        [--modalities geophysics geochemistry geology]

Outputs
-------
    outputs/checkpoints/single_<modality>_seed42_best.pt
    outputs/results/single_modality_ablation.csv
    outputs/results/single_modality_ablation.json
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.training.train import train_one_split
from src.training.utils import device_auto, load_config, write_json

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_max.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs (helpful for overnight CPU).")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override training.early_stopping_patience.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modalities", nargs="+",
                        default=["geophysics", "geochemistry", "geology"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    print(f"device: {device}")

    splits_dir = ROOT / cfg["paths"]["splits_dir"]
    results_dir = ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    split = json.load(open(splits_dir / "train_val_test.json"))
    train_idx = np.array(split["train_idx"])
    val_idx = np.array(split["val_idx"])

    # Apparently we never expect thermal here (cpu_max has use_thermal=False),
    # so the modality menu is the 3 always-present ones.
    all_modalities = ["geophysics", "geochemistry", "geology"]

    rows = []
    raw_results = {}
    for keep in args.modalities:
        if keep not in all_modalities:
            print(f"[skip] unknown modality {keep}")
            continue

        sm_cfg = copy.deepcopy(cfg)
        sm_cfg["modalities"]["drop"] = [m for m in all_modalities if m != keep]
        # Contrastive loss is undefined with one modality — kill it.
        sm_cfg["training"]["contrastive_weight"] = 0.0
        # Optional overrides (default to the config values).
        if args.epochs is not None:
            sm_cfg["training"]["epochs"] = args.epochs
            sm_cfg["training"]["scheduler_T_max"] = args.epochs
        if args.patience is not None:
            sm_cfg["training"]["early_stopping_patience"] = args.patience

        prefix = f"single_{keep}_"
        print(f"\n=== single-modality: keep={keep} "
              f"(drop={sm_cfg['modalities']['drop']}) ===")
        t0 = time.time()
        r, ckpt = train_one_split(
            sm_cfg, train_idx, val_idx, seed=args.seed, device=device,
            log_prefix=prefix,
        )
        dt = time.time() - t0
        best = r["best"]
        print(f"  done in {dt/60:.1f} min  AUROC={best['auroc']:.4f}  "
              f"AUPRC={best.get('auprc', float('nan')):.4f}")
        rows.append({
            "modality_kept": keep,
            "auroc": best["auroc"],
            "auprc": best.get("auprc", float("nan")),
            "f1": best.get("f1", float("nan")),
            "best_epoch": best["epoch"],
            "checkpoint": str(ckpt),
            "wallclock_min": dt / 60.0,
        })
        raw_results[keep] = r

    df = pd.DataFrame(rows)
    df = df.sort_values("auroc", ascending=False)
    out_csv = results_dir / "single_modality_ablation.csv"
    df.to_csv(out_csv, index=False)
    write_json(results_dir / "single_modality_ablation.json", raw_results)

    print("\nFinal table (best fold = highest AUROC):")
    print(df.to_string(index=False))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
