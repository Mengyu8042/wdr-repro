"""Figure 2: linear Gaussian grid, time-averaged log RMSE and log ES.

Reads `output/fig2_lg_grid/runs.npz` (see `summarize_fig2_runs.py`) and
writes `output/figures/fig2_lg_grid.png`.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (  # noqa: E402
    RESAMPLER_COLORS, format_kappa, npz_rows, ordered_methods, save_figure,
    setup_style,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RUNS_NPZ = PACKAGE_ROOT / "output" / "fig2_lg_grid" / "runs.npz"
OUT_PNG = PACKAGE_ROOT / "output" / "figures" / "fig2_lg_grid.png"

DIMS = [20, 50, 100]
KAPPAS = [1.0, 10.0, 50.0]
BLOCKS = [("rmse", "log(RMSE)"), ("es", "log(ES)")]

# layout, in inches
FIG_W = 6.4
PANEL = 0.67          # square panel side
GUT = 0.31            # gutter holding the y tick labels of the panel to its right
BLOCK_GAP = 0.24      # extra gap between the two metric blocks
V_GAP = 0.12          # gap between rows
PAD_L, D_LABEL = 0.02, 0.22
TOP_PAD, BLOCK_TITLE, KAPPA_TITLE = 0.07, 0.20, 0.17
XTICK, XLABEL, LEG_GAP, LEGEND, PAD_B = 0.12, 0.17, 0.04, 0.20, 0.03
FIG_H = (TOP_PAD + BLOCK_TITLE + KAPPA_TITLE + len(DIMS) * PANEL
         + (len(DIMS) - 1) * V_GAP + XTICK + XLABEL + LEG_GAP + LEGEND + PAD_B)


def load_runs() -> dict:
    """(d, kappa) -> metric -> method -> particle size -> per-replication values."""
    runs: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for r in npz_rows(RUNS_NPZ):
        key = (int(r["d"]), float(r["kappa"]))
        for metric, col in (("rmse", "log_rmse"), ("es", "log_es")):
            runs[key][metric][str(r["method"])][int(r["m"])].append(float(r[col]))
    return runs


def curves_from_results(by_metric: dict, metric: str):
    """Rebuild (x, mean, sd) per method from the per-replication values."""
    by_method = by_metric[metric]
    Ns = sorted({N for method in by_method for N in by_method[method]})
    curves = {}
    for method in ordered_methods(sorted(by_method)):
        valid_ns, means, stds = [], [], []
        for N in Ns:
            values = by_method[method].get(N)
            if not values:
                continue
            per_rep = np.asarray(values, dtype=float)
            means.append(per_rep.mean())
            stds.append(per_rep.std(ddof=1) if per_rep.size > 1 else 0.0)
            valid_ns.append(N)
        if valid_ns:
            curves[method] = (valid_ns, np.asarray(means), np.asarray(stds))
    return Ns, curves


def main() -> None:
    setup_style()
    plt.rcParams.update({"xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.titlesize": 8, "axes.labelsize": 8})
    runs = load_runs()

    nk = len(KAPPAS)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    x0 = PAD_L + D_LABEL + GUT
    y_top = FIG_H - TOP_PAD - BLOCK_TITLE - KAPPA_TITLE
    handles, axes = {}, {}

    for b, (metric, _) in enumerate(BLOCKS):
        for i, d in enumerate(DIMS):
            for j, kappa in enumerate(KAPPAS):
                col = b * nk + j
                x = x0 + col * (PANEL + GUT) + (BLOCK_GAP if b == 1 else 0.0)
                y = y_top - (i + 1) * PANEL - i * V_GAP
                ax = fig.add_axes([x / FIG_W, y / FIG_H, PANEL / FIG_W, PANEL / FIG_H])
                axes[(b, i, j)] = ax
                Ns, curves = curves_from_results(runs[(d, kappa)], metric)
                for method in curves:
                    x_vals, y_vals, y_stds = curves[method]
                    color = RESAMPLER_COLORS.get(method, "k")
                    line = ax.plot(x_vals, y_vals, "o-", lw=1.2, ms=2.6, color=color, label=method)[0]
                    ax.fill_between(x_vals, y_vals - y_stds, y_vals + y_stds,
                                    color=color, alpha=0.2, linewidth=0)
                    handles.setdefault(method, line)
                ax.set_xscale("log", base=2)
                ax.set_xticks([256, 1024, 4096])
                ax.set_xticks(Ns, minor=True)
                ax.set_xticklabels([], minor=True)
                ax.tick_params(axis="x", which="minor", length=1.5)
                ax.tick_params(axis="both", which="major", length=2.5, pad=1.5)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.grid(False)
                if i == 0:
                    ax.set_title(rf"$\kappa$={format_kappa(kappa)}", pad=3)
                if i == len(DIMS) - 1:
                    ax.set_xticklabels(["256", "1024", "4096"])
                    ax.set_xlabel("$m$", labelpad=0)
                else:
                    ax.set_xticklabels([])
                if b == 0 and j == 0:
                    ax.set_ylabel(rf"$d={d}$", labelpad=2)

    for b, (_, title) in enumerate(BLOCKS):
        left = axes[(b, 0, 0)].get_position()
        right = axes[(b, 0, nk - 1)].get_position()
        fig.text((left.x0 + right.x1) / 2.0, 1.0 - (TOP_PAD + 0.5 * BLOCK_TITLE) / FIG_H,
                 title, ha="center", va="center", fontsize=9)

    fig.legend([handles[m] for m in ordered_methods(handles.keys())],
               ordered_methods(handles.keys()),
               loc="lower center", bbox_to_anchor=(0.5, (PAD_B * 0.2) / FIG_H),
               ncol=max(1, len(handles)), frameon=False, fontsize=8,
               handlelength=1.5, columnspacing=1.5)
    save_figure(fig, OUT_PNG)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
