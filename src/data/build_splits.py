"""Build leave-one-field-out cross-validation folds and a random 70/15/15 split.

The LOF-CV split places ALL positive cells from one field in the held-out test
fold, ensuring that positives in train and test never come from the same
geothermal system — the strongest test of generalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    processed = ROOT / cfg["paths"]["processed_dir"]
    splits_dir = ROOT / cfg["paths"]["splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(processed / "labels.npy")
    train_mask = np.load(processed / "train_mask.npy")
    field_assignment = np.load(processed / "field_assignment.npy")

    labeled = np.flatnonzero(train_mask)
    n_labeled = len(labeled)

    # ---- Field clustering ---------------------------------------------------
    # Two known fields within 10 km are functionally the same hydrothermal
    # system (e.g. Brady's / Desert Peak). Without clustering, holding out
    # field A would leave field B's positive halo in the training set —
    # silent label leakage. We group fields by single-link clustering at a
    # 10 km threshold and treat each cluster as one LOF fold.
    import pandas as pd
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    fields = pd.read_csv(ROOT / "data/metadata/known_fields_details.csv") \
        if (ROOT / "data/metadata/known_fields_details.csv").exists() else None

    if fields is not None and len(fields) > 1:
        lat0 = float(fields["lat"].mean())
        cos_lat0 = float(np.cos(np.radians(lat0)))
        xy = np.column_stack([
            fields["lon"].values * 111.32 * cos_lat0,
            fields["lat"].values * 111.32,
        ])
        Z = linkage(pdist(xy), method="single")
        field_cluster = fcluster(Z, t=10.0, criterion="distance")
        # map: field_id -> cluster_id
        field_to_cluster = {fid: int(cid)
                            for fid, cid in zip(fields["field_id"].values,
                                                field_cluster)}
    else:
        field_to_cluster = {}

    # Replace field_assignment with cluster_assignment for fold construction.
    cluster_assignment = np.full_like(field_assignment, -1)
    for fid in np.unique(field_assignment):
        if fid < 0:
            continue
        cluster_assignment[field_assignment == fid] = \
            field_to_cluster.get(int(fid), int(fid))

    # ---- LOF-CV ------------------------------------------------------------
    unique_clusters = sorted(set(int(c) for c in cluster_assignment[labeled] if c >= 0))
    print(f"Building LOF-CV with {len(unique_clusters)} field-clusters "
          f"(from {len(np.unique(field_assignment[field_assignment >= 0]))} raw fields) ...")
    folds = []
    for cluster_id in unique_clusters:
        test_idx = np.flatnonzero(
            train_mask & (cluster_assignment == cluster_id)
        ).tolist()
        # Also include some negatives in the test fold, drawn at random
        neg_pool = np.flatnonzero(train_mask & (labels == 0))
        rng = np.random.default_rng(cluster_id)
        test_neg = rng.choice(neg_pool, size=min(len(neg_pool) // 5, 500),
                              replace=False).tolist()
        test_idx = sorted(set(test_idx + test_neg))
        train_idx = sorted(set(labeled.tolist()) - set(test_idx))
        folds.append({
            "field_id": cluster_id,
            "train_idx": train_idx,
            "test_idx": test_idx,
        })

    with open(splits_dir / "lof_cv_folds.json", "w") as f:
        json.dump(folds, f)
    print(f"Wrote lof_cv_folds.json with {len(folds)} folds")

    # ---- Random 70/15/15 development split --------------------------------
    rng = np.random.default_rng(42)
    perm = rng.permutation(labeled)
    n_train = int(0.7 * n_labeled)
    n_val = int(0.15 * n_labeled)
    split = {
        "train_idx": perm[:n_train].tolist(),
        "val_idx": perm[n_train:n_train + n_val].tolist(),
        "test_idx": perm[n_train + n_val:].tolist(),
    }
    with open(splits_dir / "train_val_test.json", "w") as f:
        json.dump(split, f)
    print("Wrote train_val_test.json")


if __name__ == "__main__":
    main()
