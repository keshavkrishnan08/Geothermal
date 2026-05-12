#!/usr/bin/env bash
# End-to-end pipeline — DISCOVERY-FIRST ordering.
#
# The headline output is the ranked candidate-site list with MWe estimates
# and the blind-validation result against the Zanskar Big Blind site. The
# benchmark and ablation tables are supporting evidence for the discovery
# claim — they run AFTER the discovery, not before it.
set -euo pipefail
cd "$(dirname "$0")"

# --- Phase A: data ---------------------------------------------------------
bash 01_download_data.sh
bash 02_process_data.sh

# --- Phase B: training (discovery model) -----------------------------------
# Train the model that will produce the headline continental map.
# Single seed, all labelled data — this is the model we discover with.
python -m src.training.train --config ../configs/default.yaml --random --seed 42

# --- Phase C: discovery (HEADLINE) -----------------------------------------
bash 06_discovery.sh
python -m src.evaluation.mwe_estimation --config ../configs/default.yaml
python -m src.evaluation.blind_validation --config ../configs/default.yaml

# --- Phase D: benchmarking (supports the discovery claim) ------------------
bash 03_train_dev.sh        # 5-fold smoke test of GeoProspectNet + baselines
bash 04_train_full.sh       # full LOF-CV against baselines incl. Mordensky 2025 capture rates

# --- Phase E: methodological evidence (architecture justifications) --------
bash 05_ablation.sh
python -m src.evaluation.cross_basin --config ../configs/default.yaml \
    --train_provinces Basin_and_Range --test_provinces Cascades Snake_River_Plain
bash 07_analysis.sh

# --- Phase F: figures + tables ---------------------------------------------
bash 08_figures.sh
