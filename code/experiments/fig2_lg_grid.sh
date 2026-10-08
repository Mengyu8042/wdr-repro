#!/usr/bin/env bash
# Figure 2: linear Gaussian model over dimensions and condition numbers.
# Results are written to ./outputs_lg/ and the summary to
# ./output/fig2_lg_grid/.
set -euo pipefail
PY=${PY:-python}
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"

for d in 20 50 100; do
  for kappa in 1 10 50; do
    $PY code/simulation/run_lg_grid.py \
        --d "$d" --kappa "$kappa" \
        --Ns 256,512,1024,2048,4096 --Ts 1000 --rep 20 \
        --R 3 --cv 2
  done
done

$PY code/plotting/summarize_fig2_runs.py --source outputs_lg
