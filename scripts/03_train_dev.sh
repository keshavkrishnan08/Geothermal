#!/usr/bin/env bash
# Step 3: Quick development training — 5 LOF folds + 4 flat baselines.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.training.train --config configs/default.yaml --quick
python -m src.training.train_baselines --config configs/default.yaml --quick --baseline all
