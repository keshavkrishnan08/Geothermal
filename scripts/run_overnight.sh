#!/usr/bin/env bash
# Overnight CPU pipeline — runs everything that's missing, end-to-end.
#
# Phases (sequential, each idempotent):
#   1. pytest integration suite (fail fast if data/checkpoints are broken)
#   2. True single-modality ablation training (3 models × ~20 min)
#   3. East-coast fine-tune + LOSO + MC-Dropout + cluster + MWe + pre-reg
#   4. Re-derive continental discovery from the canonical checkpoint
#      (this also regenerates the pre-registration manifest with the
#      seeded MC-Dropout, so the SHA stops drifting)
#   5. Regenerate all 20 figures
#   6. Smoke test (sanity)
#   7. Final manifest summary
#
# Each phase writes to outputs/overnight.log. Output is also tee-ed to stdout
# so you can `tail -f outputs/overnight.log`.
#
# Usage:
#   bash scripts/run_overnight.sh            # run everything
#   PHASES="3,5"  bash scripts/run_overnight.sh   # run a subset
#   SKIP_LOSO=1  bash scripts/run_overnight.sh    # skip the slow LOSO probe
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
LOG="${ROOT}/outputs/overnight.log"
mkdir -p outputs

# All phases by default; override with PHASES=1,2,4 etc.
PHASES="${PHASES:-1,2,3,4,5,6,7}"
SKIP_LOSO="${SKIP_LOSO:-0}"

# Pretty banner
banner () {
    echo "" | tee -a "$LOG"
    echo "================================================================" | tee -a "$LOG"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')  ::  $*" | tee -a "$LOG"
    echo "================================================================" | tee -a "$LOG"
}

run_phase () {
    local id="$1"; local title="$2"; shift 2
    if ! [[ ",${PHASES}," == *",${id},"* ]]; then
        echo "[skip phase ${id}]  ${title}" | tee -a "$LOG"
        return 0
    fi
    banner "PHASE ${id}: ${title}"
    local t0=$(date +%s)
    if "$@" 2>&1 | tee -a "$LOG"; then
        local dt=$(($(date +%s) - t0))
        banner "PHASE ${id} OK (${dt}s)"
    else
        local dt=$(($(date +%s) - t0))
        banner "PHASE ${id} FAILED after ${dt}s — continuing"
        return 1
    fi
}

# ---------------------------------------------------------------------------
phase1 () {
    python -m pytest tests/ -q --tb=short
}

phase2 () {
    # 40 epochs is enough on CPU; cpu_max went to ~40 itself before early stop.
    python -m src.training.single_modality_ablation \
        --config configs/cpu_max.yaml \
        --epochs 40 --patience 8 --seed 42 \
        --modalities geophysics geochemistry geology
}

phase3 () {
    local extra=""
    if [ "$SKIP_LOSO" = "1" ]; then extra="--skip_loso"; fi
    python -m src.training.east_coast_finetune \
        --config configs/cpu_max.yaml \
        --epochs 12 --lr_scale 0.2 \
        --n_eastern_neg 800 --mc_passes 8 \
        --prob_threshold 0.6 --unc_threshold 0.20 \
        --dbscan_eps_km 15.0 --dbscan_min_samples 3 \
        $extra
}

phase4 () {
    # Re-run discovery on the canonical cpu_max checkpoint with the seeded
    # MC-Dropout fix. This regenerates consensus_discoveries.csv,
    # consensus_with_mwe_v2.csv, and preregistration_manifest.csv/.json with a
    # deterministic SHA-256.
    python -m src.evaluation.discovery --config configs/cpu_max.yaml \
        --checkpoint outputs/checkpoints/random_cpu_max_seed42.pt
    # MWe v2 + economic + pre-registration are pure analysis, fast.
    python -m src.evaluation.mwe_estimation_v2 \
        --config configs/cpu_max.yaml \
        --in_csv outputs/results/consensus_discoveries.csv \
        --out_csv outputs/results/consensus_with_mwe_v2.csv
    python -m src.evaluation.preregister \
        --config configs/cpu_max.yaml
    python -m src.evaluation.temporal_holdout
    python -m src.evaluation.mordensky_comparison || true
}

phase5 () {
    python -m src.visualization.make_figures || true
}

phase6 () {
    python scripts/smoke_test.py
}

phase7 () {
    python - <<'PY'
import json, hashlib, pandas as pd
from pathlib import Path
ROOT = Path('.')

print('OVERNIGHT SUMMARY')
print('=' * 60)

def safe_print(label, path, n=False):
    p = ROOT / path
    if not p.exists():
        print(f"  [missing]  {label}: {path}"); return
    if p.suffix == '.csv':
        df = pd.read_csv(p, comment='#') if 'preregistration' in p.name else pd.read_csv(p)
        print(f"  [ok]       {label}: {len(df)} rows  →  {path}")
    elif p.suffix == '.json':
        obj = json.load(open(p))
        print(f"  [ok]       {label}: {list(obj.keys())[:6]}…  →  {path}")
    else:
        print(f"  [ok]       {label}: {p.stat().st_size} bytes  →  {path}")

safe_print('continental discoveries',
            'outputs/results/consensus_discoveries.csv')
safe_print('continental discoveries + MWe v2',
            'outputs/results/consensus_with_mwe_v2.csv')
safe_print('pre-registration manifest (csv)',
            'outputs/results/preregistration_manifest.csv')
safe_print('pre-registration manifest (json)',
            'outputs/results/preregistration_manifest.json')
safe_print('temporal hold-out',
            'outputs/results/temporal_holdout.csv')
safe_print('mordensky comparison',
            'outputs/results/mordensky_comparison.csv')
safe_print('single-modality ablation',
            'outputs/results/single_modality_ablation.csv')
safe_print('east-coast LOSO',
            'outputs/results/east_coast_loso.csv')
safe_print('east-coast discoveries + MWe',
            'outputs/results/east_coast_discoveries_with_mwe.csv')
safe_print('east-coast pre-reg manifest',
            'outputs/results/east_coast_preregistration.json')

# Western totals
p = ROOT / 'outputs/results/consensus_with_mwe_v2.csv'
if p.exists():
    df = pd.read_csv(p)
    print()
    print(f"  Western US: {len(df)} discoveries")
    print(f"    ΣP10 = {df.mwe_p10.sum():,.0f} MWe")
    print(f"    ΣP50 = {df.mwe_p50.sum():,.0f} MWe")
    print(f"    ΣP90 = {df.mwe_p90.sum():,.0f} MWe")

# Eastern totals
pe = ROOT / 'outputs/results/east_coast_discoveries_with_mwe.csv'
if pe.exists():
    df = pd.read_csv(pe)
    if len(df):
        print()
        print(f"  Eastern US: {len(df)} discoveries")
        print(f"    ΣP10 = {df.P10_MWe.sum():,.0f} MWe")
        print(f"    ΣP50 = {df.P50_MWe.sum():,.0f} MWe")
        print(f"    ΣP90 = {df.P90_MWe.sum():,.0f} MWe")

# Single-modality
sm = ROOT / 'outputs/results/single_modality_ablation.csv'
if sm.exists():
    df = pd.read_csv(sm)
    print()
    print('  Single-modality AUROC:')
    for _, r in df.iterrows():
        print(f"    {r.modality_kept:>14s}: {r.auroc:.4f}")
PY
}

# ---------------------------------------------------------------------------
banner "OVERNIGHT START  ::  PHASES=${PHASES}  ::  SKIP_LOSO=${SKIP_LOSO}"
echo "logging to: ${LOG}"
echo ""

run_phase 1 "integration tests"         phase1
run_phase 2 "single-modality ablation"  phase2
run_phase 3 "east-coast fine-tune"      phase3
run_phase 4 "re-run discovery + pre-reg" phase4
run_phase 5 "regenerate figures"        phase5
run_phase 6 "smoke test"                phase6
run_phase 7 "summary"                   phase7

banner "OVERNIGHT DONE"
