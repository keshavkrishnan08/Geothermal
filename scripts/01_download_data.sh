#!/usr/bin/env bash
# Step 1: Download all auto-downloadable datasets.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.data.download
echo
echo "Manual-download steps (USGS GEOTHERM, SMU heat flow, NOAA springs,"
echo "Smithsonian volcanoes, SRTM elevation, Landsat thermal):"
python -m src.data.download --print-manual
