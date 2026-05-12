"""Run the 8-row ablation study (A0..A7) defined in configs/ablation.yaml."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from src.training.train import train_one_split
from src.training.utils import device_auto, load_config

ROOT = Path(__file__).resolve().parents[2]


def _apply_overrides(base: Dict, overrides: Dict[str, Any]) -> Dict:
    cfg = copy.deepcopy(base)
    for dotted, value in overrides.items():
        section, key = dotted.split(".", 1)
        cfg.setdefault(section, {})[key] = value
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ablation", default="configs/ablation.yaml")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of LOF folds per variant (default 5 = quick)")
    args = parser.parse_args()

    base = load_config(args.config)
    abl = load_config(args.ablation)
    device = device_auto()

    folds = json.load(open(ROOT / base["paths"]["splits_dir"] / "lof_cv_folds.json"))
    folds = folds[: args.n_folds]
    out_rows = []

    for variant_name, spec in abl["variants"].items():
        overrides = spec["overrides"]
        cfg = _apply_overrides(base, overrides)

        aurocs, auprcs = [], []
        for fold in folds:
            r, _ = train_one_split(
                cfg, np.array(fold["train_idx"]), np.array(fold["test_idx"]),
                seed=42, device=device,
                log_prefix=f"{variant_name}_fold{fold['field_id']}_",
            )
            aurocs.append(r["best"]["auroc"])
            auprcs.append(r["best"].get("auprc", float("nan")))
        out_rows.append({
            "variant": variant_name,
            "description": spec["description"],
            "auroc_mean": float(np.nanmean(aurocs)),
            "auroc_std": float(np.nanstd(aurocs)),
            "auprc_mean": float(np.nanmean(auprcs)),
            "auprc_std": float(np.nanstd(auprcs)),
        })
        print(f"{variant_name:24s}  AUROC={out_rows[-1]['auroc_mean']:.4f} "
              f"± {out_rows[-1]['auroc_std']:.4f}")

    out_csv = ROOT / "outputs/results/table2_ablation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
