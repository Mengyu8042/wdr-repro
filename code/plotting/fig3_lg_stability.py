"""Figure 3: between-run stability metrics for the linear Gaussian model.

Writes `output/figures/fig3_lg_stability.png`. The estimate and RMSE standard 
deviations are recomputed from the per-run values in `output/fig3_stability/runs.npz`; 
the particle energy distance is a functional of the particle clouds rather than of 
these trajectories, so it is read from `output/fig3_stability/fig3_values.npz` (see
`summarize_fig3_runs.py`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import RESAMPLER_COLORS, npz_rows, ordered_methods, save_figure, setup_style  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
VALUES_NPZ = PACKAGE_ROOT / "output" / "fig3_stability" / "fig3_values.npz"
RUNS_NPZ = PACKAGE_ROOT / "output" / "fig3_stability" / "runs.npz"
OUT_PNG = PACKAGE_ROOT / "output" / "figures" / "fig3_lg_stability.png"

METRIC_TITLES = ["Particle ED", "Estimate std", "RMSE std"]


def load_values() -> tuple[list[int], list[str], dict]:
    data: dict = {title: {} for title in METRIC_TITLES}
    methods_seen: list[str] = []
    Ns_seen: list[int] = []
    for row in npz_rows(VALUES_NPZ):
        N = int(row["N"])
        method = str(row["method"])
        data[str(row["metric"])].setdefault(N, {})[method] = float(row["value"])
        if method not in methods_seen:
            methods_seen.append(method)
        if N not in Ns_seen:
            Ns_seen.append(N)
    return sorted(Ns_seen), ordered_methods(methods_seen), data


def std_ddof1(a, axis=0):
    return np.std(a, axis=axis, ddof=1) if a.shape[axis] > 1 else np.zeros(
        a.shape[:axis] + a.shape[axis + 1:])


def load_runs():
    """(method, particle size) -> arrays of the per-run values over time."""
    from collections import defaultdict
    rmse = defaultdict(lambda: defaultdict(list))
    spread = defaultdict(lambda: defaultdict(list))
    for row in npz_rows(RUNS_NPZ):
        key = (str(row["method"]), int(row["m"]))
        rmse[key][int(row["run"])].append(float(row["rmse"]))
        spread[key][int(row["run"])].append(float(row["spread"]))
    out = {}
    for key in rmse:
        out[key] = (np.asarray([rmse[key][r] for r in sorted(rmse[key])], dtype=float),
                    np.asarray([spread[key][r] for r in sorted(spread[key])], dtype=float))
    return out


def metrics_from_runs(data: dict, dimension: int = 50) -> None:
    """Recompute the two standard deviations from the per-run values."""
    if not RUNS_NPZ.exists():
        return
    for (method, N), (rmse_rt, spread_rt) in load_runs().items():
        runs = rmse_rt.shape[0]
        dispersion_t = np.sqrt((spread_rt ** 2).sum(axis=0) / (dimension * (runs - 1)))
        rmse_std_t = std_ddof1(rmse_rt, axis=0)
        data["Estimate std"][N][method] = float(np.mean(dispersion_t))
        data["RMSE std"][N][method] = float(np.mean(rmse_std_t))


def main() -> None:
    setup_style()
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": False,
        }
    )
    N_list, methods, data = load_values()
    metrics_from_runs(data)

    fig, axes = plt.subplots(1, 3, figsize=(5.20, 2.35))
    handles = {}
    for ax, title in zip(axes, METRIC_TITLES):
        metric_map = data[title]
        for method in methods:
            y_vals = [metric_map[N][method] for N in N_list]
            line = ax.plot(
                N_list,
                y_vals,
                "o-",
                lw=1.2,
                ms=2.8,
                color=RESAMPLER_COLORS.get(method, "k"),
                label=method,
            )[0]
            handles.setdefault(method, line)
        ax.set_title(title)
        ax.set_xlabel("$m$")
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_list)
        ax.set_xticklabels([str(N) for N in N_list])
        ax.grid(False)
        ax.set_box_aspect(1.0)

    fig.legend(
        [handles[m] for m in methods if m in handles],
        [m for m in methods if m in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=max(1, len(handles)),
        frameon=False,
        fontsize=8,
        handlelength=1.5,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    save_figure(fig, OUT_PNG)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
