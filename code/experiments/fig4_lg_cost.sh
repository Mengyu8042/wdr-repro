#!/usr/bin/env bash
# Figure 4: performance versus computing cost on the linear Gaussian model.
# Writes one file per run into ./output/fig4_cost/runs/. The figure script
# prefers that directory over the runs.npz table shipped with the archive.
set -euo pipefail
PY=${PY:-python}
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"
OUT="output/fig4_cost/runs"
mkdir -p "$OUT"

for mult in 1 10 50 100; do
  for N in 256 512 1024 2048; do
    for method in Multinomial Stratified Residual Hilbert WDR; do
      $PY code/simulation/run_lg_cost.py \
          --mult "$mult" --N "$N" --method "$method" --n-runs 20 --out "$OUT"
    done
  done
done
