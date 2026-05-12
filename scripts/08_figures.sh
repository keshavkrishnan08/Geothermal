#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.visualization.study_region_map
python -m src.visualization.model_comparison
python -m src.visualization.prospectivity_map
python -m src.visualization.attention_plots
python -m src.visualization.spatial_plots
python -m src.visualization.embedding_plots
python -m src.visualization.ablation_table
python -m src.visualization.discovery_table
