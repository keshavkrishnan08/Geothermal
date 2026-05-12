#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.evaluation.modality_analysis
python -m src.evaluation.spatial_analysis --n_folds 3
python -m src.evaluation.embedding_viz
python -m src.evaluation.statistical_tests
