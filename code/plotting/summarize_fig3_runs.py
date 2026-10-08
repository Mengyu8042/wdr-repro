"""Summarize the Figure 3 between-run stability results.

Reads the `results.pkl` written by the between-run stability study and writes
two files into `output/fig3_stability/`, both read by `fig3_lg_stability.py`:

  * `fig3_values.npz`, one scalar per (metric, particle size, method);
  * `runs.npz`, one row per run and timestep holding the error of that run
    against the Kalman reference and its distance to the mean of the runs.
    
Usage (after running the experiments):
    python summarize_fig3_runs.py --results-pkl <path>
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUT_NPZ = PACKAGE_ROOT / "output" / "fig3_stability" / "fig3_values.npz"


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return _NumpyCompatUnpickler(handle).load()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-pkl", type=Path, required=True,
                    help="results.pkl written by run_lg_stability.py")
    ap.add_argument("--out", type=Path, default=OUT_NPZ,
                    help="output npz path (runs.npz is written next to it)")
    args = ap.parse_args()

    payload = load_pickle(args.results_pkl)
    N_list = payload["N_list"]
    methods = list(payload["results_all"][N_list[0]].keys())

    args.out.parent.mkdir(parents=True, exist_ok=True)

    ref = np.asarray(payload["ref_path"], dtype=float)
    runs_path = args.out.parent / "runs.npz"
    run_rows = []
    for N in N_list:
        for method in methods:
            runs = np.asarray(
                payload["per_run_estimates"][N][method]["est_runs"], dtype=float)
            rmse_rt = np.sqrt(np.mean((runs - ref[None, :, :]) ** 2, axis=2))
            spread_rt = np.linalg.norm(
                runs - runs.mean(axis=0, keepdims=True), axis=2)
            for r in range(runs.shape[0]):
                for t_idx in range(runs.shape[1]):
                    run_rows.append((method, N, r + 1, t_idx + 1,
                                     float(rmse_rt[r, t_idx]),
                                     float(spread_rt[r, t_idx])))
    run_cols = ["method", "m", "run", "t", "rmse", "spread"]
    np.savez_compressed(runs_path, **{c: np.asarray([row[i] for row in run_rows])
                                      for i, c in enumerate(run_cols)})
    print(f"wrote {runs_path}")

    val_rows = []
    for N in N_list:
        for method in methods:
            s = payload["results_all"][N][method]
            val_rows.append(("Particle ED", N, method, float(s["br_ed"])))
            val_rows.append(("Estimate std", N, method, float(s["est_dispersion_mean"])))
            val_rows.append(("RMSE std", N, method, float(s["rmse_std_mean"])))
    val_cols = ["metric", "N", "method", "value"]
    np.savez_compressed(args.out, **{c: np.asarray([row[i] for row in val_rows])
                                     for i, c in enumerate(val_cols)})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
