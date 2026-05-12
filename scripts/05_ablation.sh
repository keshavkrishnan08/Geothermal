#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.evaluation.ablation --config configs/default.yaml --ablation configs/ablation.yaml --n_folds 5
python -m src.visualization.ablation_table
