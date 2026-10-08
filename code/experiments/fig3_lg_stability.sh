#!/usr/bin/env bash
# Figure 3: between-run stability on one linear Gaussian dataset.
# Results are written to ./outputs_between_run/ and the summary to
# ./output/fig3_stability/.
set -euo pipefail
PY=${PY:-python}
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"

$PY code/simulation/run_lg_stability.py \
    --d 50 \
    --N 256,512,1024,2048 --n-runs 20 --T 100 --R 3 --cv 2

RUN_DIR=$(ls -d outputs_between_run/LG_d50_R3/results_* | tail -1)
$PY code/plotting/summarize_fig3_runs.py --results-pkl "$RUN_DIR/results.pkl"
