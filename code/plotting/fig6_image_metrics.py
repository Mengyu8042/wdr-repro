"""Figure 6: conditional image generation metrics versus wall-clock cost.

Reads `output/fig6_image/fig6_values.npz` and writes
`output/figures/fig6_image_metrics_vs_cost.png`: one row per dataset
(MNIST, CIFAR-10), one column per metric, one shared legend. 
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
VALUES_NPZ = PACKAGE_ROOT / "output" / "fig6_image" / "fig6_values.npz"
OUT_PNG = PACKAGE_ROOT / "output" / "figures" / "fig6_image_metrics_vs_cost.png"

DATASETS = ["MNIST", "CIFAR-10"]

METHOD_STYLE = {
    "Vanilla TDS": {"color": "#8D8D8D", "linestyle": "-"},
    "TDS-WDR": {"color": "#e31a1c", "linestyle": "-"},
    "TDS-T-WDR": {"color": "#972d2f", "linestyle": "-"},
}

PLOT_METHODS = ["Vanilla TDS", "TDS-WDR", "TDS-T-WDR"]

# (mean column, standard-error column, panel title)
METRIC_PANELS = [
    ("fid", None, "FID↓"),
    ("acc", "acc_se", "Accuracy↑"),
    ("lpips", "lpips_se", "LPIPS diversity↑"),
    ("between_over_within", "between_over_within_se", "Between-within ratio↓"),
]

M_MARKERS = {16: "o", 32: "s", 64: "^", 128: "v"}

WIDTH_IN = 5.50          # final on-page width


def load_rows() -> list[dict]:
    rows = []
    z = np.load(VALUES_NPZ, allow_pickle=False)
    cols = list(z.files)
    for vals in zip(*(z[c] for c in cols)):
        row = dict(zip(cols, vals))
        parsed = {k: (str(v) if getattr(v, "dtype", None) is not None
                      and v.dtype.kind == "U" else v) for k, v in row.items()}
        parsed["particles"] = int(row["particles"])
        for col in ("fid", "acc", "acc_se", "lpips", "lpips_se",
                    "between_over_within", "between_over_within_se", "cost_s"):
            parsed[col] = float(row[col]) if str(row.get(col, "")) != "" else None
        rows.append(parsed)
    return rows


def main() -> None:
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
        "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": False,
    })
    rows = load_rows()

    w = WIDTH_IN
    fig, axes = plt.subplots(2, 4, figsize=(w, 0.68 * w), squeeze=False)
    handles, seen_ms = {}, set()
    for r, dataset in enumerate(DATASETS):
        ds_rows = [x for x in rows if x["dataset"] == dataset]
        for c, (mean_col, se_col, _) in enumerate(METRIC_PANELS):
            ax = axes[r][c]
            for method in PLOT_METHODS:
                style = METHOD_STYLE[method]
                subset = sorted((x for x in ds_rows if x["method"] == method),
                                key=lambda x: x["particles"])
                points = []
                for x in subset:
                    y = x.get(mean_col)
                    if y is None or x.get("cost_s") is None:
                        continue
                    points.append((x["cost_s"], y,
                                   x.get(se_col) if se_col else None,
                                   x["particles"]))
                if not points:
                    continue
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                line = ax.plot(xs, ys, color=style["color"],
                               linestyle=style["linestyle"], lw=1.3,
                               label=method)[0]
                handles.setdefault(method, line)
                for x_, y_, _, m in points:
                    ax.plot(x_, y_, marker=M_MARKERS.get(m, "o"),
                            color=style["color"], linestyle="", ms=3.5,
                            markeredgecolor="black", markeredgewidth=0.3)
                    seen_ms.add(m)
                if se_col is not None:
                    se_pts = [(x_, y_, s) for x_, y_, s, _ in points
                              if s is not None]
                    if len(se_pts) >= 2:
                        ax.fill_between([p[0] for p in se_pts],
                                        [p[1] - p[2] for p in se_pts],
                                        [p[1] + p[2] for p in se_pts],
                                        color=style["color"], alpha=0.18,
                                        linewidth=0)
            ax.set_box_aspect(1.0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(length=2.5, pad=1.0)
            # metric titles are drawn after tight_layout as free-floating
            # text, so the layout engine does not widen columns for them
            if r == 1:
                ax.set_xlabel("Cost (s)", labelpad=1)

    method_order = [m for m in PLOT_METHODS if m in handles]
    leg1 = fig.legend([handles[m] for m in method_order], method_order,
                      loc="lower center", bbox_to_anchor=(0.5, 0.058),
                      ncol=len(method_order), frameon=False,
                      columnspacing=1.2, handlelength=1.4, handletextpad=0.5)
    fig.add_artist(leg1)
    m_order = sorted(seen_ms)
    mh = [Line2D([], [], marker=M_MARKERS.get(m, "o"), color="gray",
                 linestyle="", ms=3.5, markeredgecolor="black",
                 markeredgewidth=0.3) for m in m_order]
    fig.legend(mh, [fr"$m$={m}" for m in m_order], loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=len(m_order), frameon=False,
               columnspacing=1.0, handletextpad=0.3)

    fig.tight_layout(rect=[0, 0.155, 1, 1], h_pad=3.2, w_pad=1.15)
    # metric titles: centered over each top-row panel, outside the layout
    title_texts = []
    for c, (_, _, label) in enumerate(METRIC_PANELS):
        box = axes[0][c].get_position()
        title_texts.append(fig.text(0.5 * (box.x0 + box.x1), box.y1 + 0.012,
                                    label, ha="center", va="bottom",
                                    fontsize=8.5))
    # keep titles inside the canvas, so bbox_inches="tight" does not spend
    # width on their spill
    fig.canvas.draw()
    for t in title_texts:
        bb = t.get_window_extent().transformed(fig.transFigure.inverted())
        x, y = t.get_position()
        if bb.x1 > 0.998:
            t.set_position((x - (bb.x1 - 0.998), y))
        if bb.x0 < 0.002:
            t.set_position((x + (0.002 - bb.x0), y))
    # nudge adjacent titles apart if they crowd (left one moves left)
    fig.canvas.draw()
    MIN_GAP = 0.012
    for t_left, t_right in zip(title_texts, title_texts[1:]):
        bl = t_left.get_window_extent().transformed(fig.transFigure.inverted())
        br = t_right.get_window_extent().transformed(fig.transFigure.inverted())
        overlap = bl.x1 + MIN_GAP - br.x0
        if overlap > 0:
            x, y = t_left.get_position()
            t_left.set_position((x - overlap, y))
    # bold dataset titles centered over each row, above the panel titles
    for r, dataset in enumerate(DATASETS):
        boxes = [axes[r][c].get_position() for c in range(4)]
        x_mid = 0.5 * (boxes[0].x0 + boxes[-1].x1)
        y_top = max(b.y1 for b in boxes)
        fig.text(x_mid, y_top + (0.058 if r == 0 else 0.028), dataset,
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
