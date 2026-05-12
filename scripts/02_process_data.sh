#!/usr/bin/env bash
# Step 2: Process all four modalities, build labels, splits, neighbors.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.data.process_geophysics
python -m src.data.process_geochemistry
python -m src.data.process_thermal
python -m src.data.process_geology
python -m src.data.build_labels
python -m src.data.build_splits
python -m src.data.build_neighbors
