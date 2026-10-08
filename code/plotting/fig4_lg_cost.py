"""Figure 4: linear Gaussian performance--cost analysis.

Reads the run results in `output/fig4_cost/` (RMSE, energy score, and energy
distance of every run; the recorded computing times) and writes
`output/figures/fig4_lg_cost.png`: one row per metric, one column per
sub-step multiplier.
"""
import argparse
import collections
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter
import numpy as np

ORDER = ["Multinomial", "Stratified", "Residual", "Hilbert", "WDR"]
COLOR = {"Multinomial": "#8D8D8D", "Stratified": "#a6cee3",
         "Residual": "#1f78b4", "Hilbert": "#fdbf6f", "WDR": "#e31a1c"}
N_MARKERS = {256: "o", 512: "s", 1024: "^", 2048: "D"}
HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TIMES_NPZ = os.path.join(PACKAGE_ROOT, "output", "fig4_cost", "computing_times.npz")
sys.path.insert(0, HERE)
from figstyle import npz_rows  # noqa: E402


def load_times(times_npz):
    """(sub-steps, N, method) -> the recorded propagation and resampling times."""
    table = {}
    for r in npz_rows(times_npz):
        table[(int(r["sub_steps"]), int(r["N"]), str(r["method"]))] = {
            "prop_s": float(r["propagation_s"]),
            "resamp_s": float(r["resampling_s"])}
    return table


def load_runs(path):
    """One record per run, from the summary table or a directory of run files."""
    if os.path.isdir(path):
        return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(path, "*.json")))]
    return npz_rows(path)


def load_cells(rows, key):
    """(mult, N, method) -> list of the values stored under `key`."""
    vals = collections.defaultdict(list)
    for d in rows:
        value = d.get(key)
        if value in (None, ""):
            continue
        vals[(int(d["mult"]), int(d["N"]), d["resampler"])].append(float(value))
    return vals


def series(metric, mult, meth, Ns, vals, log):
    """(Ns, xs, ys, lo, hi) for one method in one panel, or None."""
    ns, xs, ys, err = [], [], [], []
    for N in Ns:
        k = (mult, N, meth)
        if k not in log:
            continue
        x = log[k]["prop_s"] + log[k]["resamp_s"]
        if k not in vals[metric]:
            continue
        v = np.asarray(vals[metric][k], dtype=float)
        y, e = v.mean(), v.std(ddof=1)
        ns.append(N); xs.append(x); ys.append(y); err.append(e)
    if not xs:
        return None
    return ns, np.array(xs), np.array(ys), np.array(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="run summary csv, or a directory of per-run files "
                         "(default: the rerun directory if present, else the "
                         "summary csv)")
    ap.add_argument("--outdir", default=os.path.join(PACKAGE_ROOT, "output", "figures"),
                    help="directory the figure is written to")
    ap.add_argument("--multipliers", default="1,10,50,100",
                    help="comma-separated sub-step multipliers to plot")
    ap.add_argument("--dump", default="",
                    help="optional npy path to dump the plotted values")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
        "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": False,
    })
    log = load_times(TIMES_NPZ)
    root = args.root
    if root is None:
        rerun = os.path.join(PACKAGE_ROOT, "output", "fig4_cost", "runs")
        root = rerun if os.path.isdir(rerun) else os.path.join(
            PACKAGE_ROOT, "output", "fig4_cost", "runs.npz")
    rows = load_runs(root)
    vals = {"rmse": load_cells(rows, "rmse"),
            "es": load_cells(rows, "es"),
            "ed": load_cells(rows, "ed")}
    mults = [int(x) for x in args.multipliers.split(",")]
    Ns = sorted({k[1] for k in vals["rmse"]})
    rows = [("rmse", "RMSE"), ("es", "ES"), ("ed", "ED")]

    W = 5.28
    LEG_IN = 0.63
    H = (5.27 - LEG_IN) * (W / 6.28) + LEG_IN
    fig, axes = plt.subplots(len(rows), len(mults), figsize=(W, H),
                             sharey="row", sharex="col", squeeze=False)
    handles, seen_ns, dumped = {}, set(), {}
    for r, (metric, ylabel) in enumerate(rows):
        for c, mult in enumerate(mults):
            ax = axes[r][c]
            for meth in ORDER:
                out = series(metric, mult, meth, Ns, vals, log)
                if out is None:
                    continue
                ns, xs, ys, err = out
                dumped[(metric, mult, meth)] = (xs, ys, err)
                line = ax.plot(xs, ys, "-", lw=1.4, color=COLOR[meth], label=meth)[0]
                ax.fill_between(xs, ys - err, ys + err, color=COLOR[meth],
                                alpha=0.18, linewidth=0)
                for x, y, N in zip(xs, ys, ns):
                    ax.plot(x, y, marker=N_MARKERS.get(N, "o"), color=COLOR[meth],
                            ms=4, linestyle="", markeredgecolor="black",
                            markeredgewidth=0.3)
                    seen_ns.add(N)
                handles.setdefault(meth, line)
            ax.xaxis.set_minor_formatter(NullFormatter())
            # small panels: two x intervals keep the tick labels apart
            ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
            ax.grid(False)
            ax.set_box_aspect(1.0)
            if r == 0:
                ax.set_title(fr"$\gamma$={mult}")
            if r == len(rows) - 1:
                ax.set_xlabel("Total cost (s)")
            else:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel(ylabel)

    meth_leg = fig.legend([handles[m] for m in ORDER if m in handles],
                          [m for m in ORDER if m in handles],
                          loc="lower center",
                          bbox_to_anchor=(0.5, 0.075 * 3.68 / H),
                          ncol=len(handles), frameon=False, fontsize=8)
    fig.add_artist(meth_leg)
    nh = [Line2D([], [], marker=N_MARKERS[n], color="gray", markeredgecolor="black",
                 markeredgewidth=0.3, linestyle="", ms=4) for n in sorted(seen_ns)]
    fig.legend(nh, [fr"$m$={n}" for n in sorted(seen_ns)], loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=len(nh), frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, LEG_IN / H, 1, 1], w_pad=1.2, h_pad=0.8)
    fig.subplots_adjust(hspace=0.28)
    base = os.path.join(args.outdir, "fig4_lg_cost")
    fig.savefig(base + ".png", bbox_inches="tight")
    plt.close(fig)
    if args.dump:
        np.save(args.dump, np.array(
            [(f"{k[0]}|{k[1]}|{k[2]}", v[0].tolist(), v[1].tolist(), v[2].tolist())
             for k, v in sorted(dumped.items(), key=lambda kv: str(kv[0]))], dtype=object),
            allow_pickle=True)
    print("wrote", base + ".png", f"({len(dumped)} curves)")


if __name__ == "__main__":
    main()
