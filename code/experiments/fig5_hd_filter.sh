#!/usr/bin/env bash
# Figure 5: block particle filter on the spatial model.
# Writes one file per run into ./output/fig5_hd/runs/; the figure script reads
# that directory directly.
set -euo pipefail
PY=${PY:-python}
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"

$PY code/simulation/run_hd_filter.py --es --ed \
    --n 45 --rho 6 --alpha 0.95 --obs-frac 0.1 \
    --block 5 --T 200 \
    --ms 256 512 1024 2048 4096 \
    --n-runs 10 \
    --methods Multinomial Stratified Residual Hilbert WDR \
    --out output/fig5_hd/runs
