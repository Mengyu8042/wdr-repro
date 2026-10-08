"""Figure 5: high-dimensional spatial model, accuracy versus computing cost.

Reads the run results in `output/fig5_hd/` (RMSE, energy score, and energy
distance of every run in `runs.npz`; the recorded computing times) and
writes `output/figures/fig5_hd_cost.png`: RMSE, energy score, and energy
distance versus total cost, side by side.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np

from hd_plot_utils import METHODS, COLOR, load, series, setup_style

MS = [256, 512, 1024, 2048, 4096]
M_MARKERS = {256: "o", 512: "s", 1024: "^", 2048: "D", 4096: "v"}
T_STEPS = 200

COST_TIMES_NPZ = "cost_times.npz"
PACKAGE_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_COST_DIR = os.path.join(PACKAGE_ROOT, "output", "fig5_hd")


def load_recorded_times(cost_dir):
    """The recorded times, or None when the file is absent."""
    from hd_plot_utils import npz_rows
    path = os.path.join(cost_dir, COST_TIMES_NPZ)
    if not os.path.exists(path):
        return None
    return npz_rows(path)


def load_measured_prop(rows):
    """Particle size -> measured propagation time per step, in milliseconds."""
    return {int(r["m"]): float(r["ms"]) for r in rows
            if r["component"] == "propagation" and int(r["n"]) == 45}

WARM_MULTS = (10, 50, 100)


def load_resampler_table(rows):
    """Median resampling cost (ms per call) per (m, method) over the warm
    sub-step settings, from the recorded times."""
    vals = {}
    for r in rows:
        if (r["component"] != "resampling" or r["method"] not in METHODS
                or int(r["sub_steps"]) not in WARM_MULTS):
            continue
        vals.setdefault((int(r["m"]), r["method"]), []).append(float(r["ms"]))
    table = {}
    for (m, meth), v in vals.items():
        table.setdefault(m, {})[meth] = float(np.median(v))
    return table


PROP_TABLE = None                       # set from load_measured_prop in main
def prop_ms(m):
    return PROP_TABLE[m]


RESAMPLER_TABLE = None                  # set in main from load_resampler_table
def resampler_ms(method, m):
    c = RESAMPLER_TABLE[m][method]
    if method == "Hilbert":
        # d-linear rescaling of the Hilbert-curve charge from the log's d=50
        # to the block dimension 25.
        c *= 25.0 / 50.0
    return c


def total_cost_s(method, m):
    return T_STEPS * (prop_ms(m) + resampler_ms(method, m)) / 1000.0


def draw(s, out_base, ylabel="RMSE"):
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    handles = {}
    for meth in METHODS:
        m_arr, lg, se = (np.array(x) for x in s[meth])
        if m_arr.size == 0:
            continue
        xs = np.array([total_cost_s(meth, int(m)) for m in m_arr])
        ys = np.exp(lg)
        line = ax.plot(xs, ys, "-", lw=1.6, color=COLOR[meth],
                       label=meth)[0]
        ax.fill_between(xs, np.exp(lg - se), np.exp(lg + se),
                        color=COLOR[meth], alpha=0.18, linewidth=0)
        for x, y, m in zip(xs, ys, m_arr):
            ax.plot(x, y, marker=M_MARKERS[int(m)], color=COLOR[meth],
                    ms=5, linestyle="", markeredgecolor="black",
                    markeredgewidth=0.3)
        handles[meth] = line
    ax.set_xlabel("Total cost (s)")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(False)

    ax.margins(y=0.06)
    legend_fontsize = plt.rcParams["axes.labelsize"]
    by = handles
    # column-major fill; these orders read row-major as the natural order
    order = [m for m in ("Multinomial", "Hilbert", "Stratified", "WDR",
                         "Residual") if m in by]
    leg1 = fig.legend([by[m] for m in order], order, loc="lower center",
                      bbox_to_anchor=(0.5, 0.155), ncol=3, frameon=False,
                      fontsize=legend_fontsize, columnspacing=1.0,
                      handlelength=1.5, handletextpad=0.5)
    fig.add_artist(leg1)
    m_order = [256, 2048, 512, 4096, 1024]
    mh = [Line2D([], [], marker=M_MARKERS[m], color="gray",
                 markeredgecolor="black", markeredgewidth=0.3,
                 linestyle="", ms=5, label=fr"$m$={m}") for m in m_order]
    fig.legend(mh, [fr"$m$={m}" for m in m_order], loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
               fontsize=legend_fontsize, columnspacing=0.8,
               handletextpad=0.3)
    fig.tight_layout(rect=(0.0, 0.32, 1.0, 1.0))
    fig.savefig(out_base + ".png", bbox_inches="tight")
    plt.close(fig)


def draw_dual(s_rmse, s_es, s_ed, out_base):
    """Main-text figure: RMSE, ES, and ED versus total cost, side by side,
    Figure-6 panel style with the shared two-row bottom legend."""
    fig, axes = plt.subplots(1, 3, figsize=(1.77 * 3, 2.10))
    handles = {}
    seen = set()
    for ax, s, ylabel in ((axes[0], s_rmse, "RMSE"), (axes[1], s_es, "ES"),
                          (axes[2], s_ed, "ED")):
        for meth in METHODS:
            m_arr, lg, se = (np.array(x) for x in s[meth])
            if m_arr.size == 0:
                continue
            xs = np.array([total_cost_s(meth, int(m)) for m in m_arr])
            ys = np.exp(lg)
            line = ax.plot(xs, ys, "-", lw=1.2, color=COLOR[meth],
                           label=meth)[0]
            ax.fill_between(xs, np.exp(lg - se), np.exp(lg + se),
                            color=COLOR[meth], alpha=0.18, linewidth=0)
            for x, y, m in zip(xs, ys, m_arr):
                ax.plot(x, y, marker=M_MARKERS[int(m)], color=COLOR[meth],
                        ms=3.5, linestyle="", markeredgecolor="black",
                        markeredgewidth=0.3)
                seen.add(int(m))
            handles.setdefault(meth, line)
        ax.set_xlabel("Total cost (s)")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.margins(y=0.06)
        ax.grid(False)
        ax.set_box_aspect(1.0)
    fs = plt.rcParams["axes.labelsize"]
    order = [m for m in METHODS if m in handles]
    leg1 = fig.legend([handles[m] for m in order], order, loc="lower center",
                      bbox_to_anchor=(0.5, 0.045), ncol=len(order),
                      frameon=False, fontsize=fs, columnspacing=0.8,
                      handlelength=1.2, handletextpad=0.4)
    fig.add_artist(leg1)
    mh = [Line2D([], [], marker=M_MARKERS[m], color="gray",
                 markeredgecolor="black", markeredgewidth=0.3,
                 linestyle="", ms=3.5) for m in sorted(seen)]
    fig.legend(mh, [fr"$m$={m}" for m in sorted(seen)], loc="lower center",
               bbox_to_anchor=(0.5, -0.065), ncol=len(mh), frameon=False,
               fontsize=fs, columnspacing=0.6, handletextpad=0.3)
    fig.tight_layout(rect=(0.0, 0.27, 1.0, 1.0), w_pad=1.6)
    fig.savefig(out_base + ".png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_base + ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="run summary npz, or a directory of per-run files "
                         "(default: the rerun directory if present, else the "
                         "summary npz)")
    ap.add_argument("--outdir", default=os.path.join(PACKAGE_ROOT, "output",
                                                     "figures"),
                    help="directory the figure is written to")
    ap.add_argument("--cost-dir", default=DEFAULT_COST_DIR,
                    help="directory holding the recorded times "
                         "(independent of --root)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    setup_style()

    global PROP_TABLE, RESAMPLER_TABLE
    recorded = load_recorded_times(args.cost_dir)
    if recorded is None:
        raise SystemExit(
            f"no recorded times found in {args.cost_dir} "
            f"(looked for {COST_TIMES_NPZ})")
    PROP_TABLE = load_measured_prop(recorded)
    RESAMPLER_TABLE = load_resampler_table(recorded)
    print(f"propagation source: {COST_TIMES_NPZ}", PROP_TABLE)
    print("resampler source: recorded per-call medians (sub-steps 10/50/100)")
    print("per-step cost table (ms): prop + resampler")
    for m in MS:
        row = "  ".join(f"{meth[:5]}={prop_ms(m) + resampler_ms(meth, m):8.1f}"
                        for meth in METHODS)
        print(f"  m={m:5d} prop={prop_ms(m):7.1f}  {row}")

    root = args.root
    if root is None:
        rerun = os.path.join(PACKAGE_ROOT, "output", "fig5_hd", "runs")
        root = rerun if os.path.isdir(rerun) else os.path.join(
            PACKAGE_ROOT, "output", "fig5_hd", "runs.npz")
    cells = load(root)
    s_rmse, _ = series(cells, "rmse_oracle")
    s_es, _ = series(cells, "es_oracle")
    s_ed, ns = series(cells, "ed")
    print("replications per particle size", ns)
    draw_dual(s_rmse, s_es, s_ed,
              os.path.join(args.outdir, "fig5_hd_cost"))


if __name__ == "__main__":
    main()
