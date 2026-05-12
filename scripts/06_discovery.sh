#!/usr/bin/env bash
# Discovery pipeline (HEADLINE). Produces:
#   outputs/maps/prospectivity.tif         continental p(geothermal) GeoTIFF
#   outputs/results/table3_discoveries.csv ranked candidate sites
#   outputs/results/discoveries_with_mwe.csv  ditto + MWe estimates
#   outputs/results/blind_validation.csv      Zanskar Big Blind percentile check
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.evaluation.discovery --config configs/default.yaml
python -m src.evaluation.mwe_estimation --config configs/default.yaml
python -m src.evaluation.blind_validation --config configs/default.yaml
python -m src.visualization.discovery_table
