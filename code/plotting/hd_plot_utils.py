"""Shared helpers for the high-dimensional spatial figure (Figure 5).

Provides the method order and colors, the loader for the per-run result
files, and the aggregation of runs into curves. `fig5_hd_cost.py` builds
the published figure on top of these helpers.
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
import numpy as np

METHODS = ["Multinomial", "Stratified", "Residual", "Hilbert", "WDR"]
COLOR = {"Multinomial": "#8D8D8D", "Stratified": "#a6cee3",
         "Residual": "#1f78b4", "Hilbert": "#fdbf6f", "WDR": "#e31a1c"}
MAJOR_TICKS = [256, 1024, 4096]


def setup_style():
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
        "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": False,
    })


from figstyle import npz_rows  # noqa: E402


def load(root):
    """(m, rep) -> method -> record, from the summary table or run files."""
    cells = defaultdict(dict)
    if os.path.isdir(root):
        records = [json.load(open(f))
                   for f in sorted(glob.glob(os.path.join(root, "*_b5_m*_rep*.json")))]
    else:
        records = npz_rows(root)
    for d in records:
        if d["method"] in METHODS:
            cells[(int(d["m"]), int(d["rep"]))][d["method"]] = d
    return cells


def series(cells, key):
    """Geometric mean and +-1 sd (of log values, across the paired
    replicates where all five methods completed) per particle size m."""
    ms = sorted({m for m, _ in cells})
    out, ns = {}, []
    for meth in METHODS:
        out[meth] = ([], [], [])
    for m in ms:
        reps = [r for mm, r in cells if mm == m
                and len(cells[(mm, r)]) == len(METHODS)
                and all(key in cells[(mm, r)][meta] for meta in METHODS)]
        if not reps:
            continue
        ns.append((m, len(reps)))
        for meth in METHODS:
            v = np.log(np.array([cells[(m, r)][meth][key] for r in reps]))
            out[meth][0].append(m)
            out[meth][1].append(float(v.mean()))
            out[meth][2].append(float(v.std(ddof=1)) if len(reps) > 1 else 0.0)
    return out, ns


def draw(s, out_base, ylabel="RMSE"):
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    handles, labels = [], []
    for meth in METHODS:
        m, lg, sd = (np.array(x) for x in s[meth])
        if m.size == 0:
            continue
        line = ax.plot(m, np.exp(lg), "o-", lw=1.8, ms=4,
                       color=COLOR[meth], label=meth)[0]
        ax.fill_between(m, np.exp(lg - sd), np.exp(lg + sd),
                        color=COLOR[meth], alpha=0.2, linewidth=0)
        handles.append(line)
        labels.append(meth)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    all_ms = sorted(set(s["WDR"][0]))
    ax.set_xticks([t for t in MAJOR_TICKS if t in all_ms])
    ax.set_xticklabels([str(t) for t in MAJOR_TICKS if t in all_ms])
    ax.set_xticks(all_ms, minor=True)
    ax.set_xticklabels([], minor=True)
    ax.tick_params(axis="x", which="minor", length=2)
    lo, hi = ax.get_ylim()
    span = hi - lo
    step = next(x for x in (0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
                if span / x <= 7)
    ticks = [t for t in np.arange(step, 200.0, step) if lo < t < hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("m")
    ax.set_ylabel(ylabel)
    by = dict(zip(labels, handles))
    order = [m for m in ("Multinomial", "Hilbert", "Stratified", "WDR",
                         "Residual") if m in by]
    fig.legend([by[m] for m in order], order, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
               columnspacing=1.0, handlelength=1.6, handletextpad=0.5)
    fig.tight_layout(rect=(0.0, 0.17, 1.0, 1.0))
    fig.savefig(out_base + ".png", bbox_inches="tight")
    plt.close(fig)
