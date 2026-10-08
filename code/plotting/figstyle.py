"""Shared plotting style for all figures in the reproducibility package."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def npz_rows(path):
    """The rows of a column-stored `.npz` table, one dict per row."""
    z = np.load(path, allow_pickle=False)
    cols = list(z.files)
    return [dict(zip(cols, vals)) for vals in zip(*(z[c] for c in cols))]

RESAMPLER_ORDER = [
    "Multinomial",
    "Stratified",
    "Residual",
    "Hilbert",
    "WDR",
    "T-WDR",
]

RESAMPLER_COLORS = {
    "Multinomial": "#8D8D8D",
    "Stratified": "#a6cee3",
    "Residual": "#1f78b4",
    "Hilbert": "#fdbf6f",
    "WDR": "#e31a1c",
    "T-WDR": "#972d2f",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 12,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": False,
        }
    )


def ordered_methods(methods: Iterable[str]) -> list[str]:
    method_set = list(dict.fromkeys(methods))
    front = [m for m in RESAMPLER_ORDER if m in method_set]
    tail = [m for m in method_set if m not in front]
    return front + tail


def format_kappa(kappa: float) -> str:
    return str(int(kappa)) if float(kappa).is_integer() else str(kappa)


def save_figure(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
