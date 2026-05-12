#!/usr/bin/env bash
# Archive outputs/ at the end of a Kaggle session so they survive the 12-hour
# session timeout. Run this as the last cell of any Kaggle notebook that
# trains models.
set -euo pipefail
cd "$(dirname "$0")/.."

# Sync only the things worth saving — not raw data (recoverable) or random
# scratch files.
tar czf /kaggle/working/geoprospect_outputs.tar.gz \
    outputs/results \
    outputs/checkpoints \
    outputs/figures \
    outputs/maps \
    2>/dev/null || true

echo "Wrote /kaggle/working/geoprospect_outputs.tar.gz"
ls -lh /kaggle/working/geoprospect_outputs.tar.gz 2>/dev/null || true
