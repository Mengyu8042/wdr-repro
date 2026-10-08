"""Summarize the Figure 2 linear Gaussian runs.

Reads the raw `results.pkl` written by `code/simulation/run_lg_grid.py` for each
(d, kappa) setting and keeps the numbers Figure 2 plots: one row per run
(setting, method, particle size m, replication) with the time average of its
log RMSE and of its log energy score, written as one `runs.npz` into
`output/fig2_lg_grid/`; `fig2_lg_grid.py` reads it.

Usage (after running the experiments):
    python summarize_fig2_runs.py --source <outputs_lg root>
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUT_NPZ = PACKAGE_ROOT / "output" / "fig2_lg_grid" / "runs.npz"

DIMS = [20, 50, 100]
KAPPAS = [1.0, 10.0, 50.0]


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load pickles across the numpy 1.x / 2.x module rename."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load_results(source: Path, d: int, kappa: float):
    cell = f"d{d}_kappa{kappa}_r1.0_R3_cv2"
    paths = [p for p in sorted(glob.glob(f"{source}/{cell}/results_*/results.pkl"))
             if "trunc" not in p]
    if not paths:
        raise SystemExit(f"missing results for {cell} under {source}")
    with open(paths[-1], "rb") as f:
        return _NumpyCompatUnpickler(f).load()


def per_rep_time_avg_log(arr) -> np.ndarray:
    """Per-replication time average of the log metric (float64)."""
    a = np.asarray(arr, dtype=float)
    return np.log(np.maximum(a, 1e-12)).mean(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="outputs_lg root holding the per-setting results")
    ap.add_argument("--out", type=Path, default=OUT_NPZ,
                    help="output npz path")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DIMS:
        for kappa in KAPPAS:
            payload = load_results(args.source, d, kappa)
            rmse, es = payload["metrics_N"]["rmse"], payload["metrics_N"]["es"]
            for method in rmse:
                for N in payload["Ns"]:
                    if N not in rmse[method]:
                        continue
                    log_rmse = per_rep_time_avg_log(rmse[method][N])
                    log_es = per_rep_time_avg_log(es[method][N])
                    for rep, (lr, le) in enumerate(zip(log_rmse, log_es), start=1):
                        rows.append(dict(d=d, kappa=int(kappa), method=method,
                                         m=int(N), replication=rep,
                                         log_rmse=float(lr), log_es=float(le)))
    rows.sort(key=lambda r: (r["d"], r["kappa"], r["method"], r["m"],
                             r["replication"]))
    cols = ["d", "kappa", "method", "m", "replication", "log_rmse", "log_es"]
    np.savez_compressed(args.out, **{c: np.asarray([r[c] for r in rows]) for c in cols})
    print(f"wrote {args.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
