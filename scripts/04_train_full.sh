#!/usr/bin/env bash
# Step 4: Full LOF-CV — GeoProspectNet + all baselines.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.training.train --config configs/default.yaml --all_folds
python -m src.training.train_baselines --config configs/default.yaml --all_folds --baseline all
python -m src.evaluation.cross_validation
