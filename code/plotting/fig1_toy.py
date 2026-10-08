"""Figure 1: toy illustration of sample impoverishment and free-support resampling.

Usage:
    python fig1_toy.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUT_PNG = PACKAGE_ROOT / "output" / "figures" / "fig1_toy.png"


def gaussian_pdf(X, mu, Sigma):
    d = mu.shape[0]
    Xc = X - mu
    Sinv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    quad = np.einsum("ni,ij,nj->n", Xc, Sinv, Xc)
    return np.exp(-0.5 * quad) / np.sqrt(((2 * np.pi) ** d) * det)


def sinkhorn(a, b, C, eps=1.0, n_iter=250):
    K = np.exp(-C / max(eps, 1e-12)) + 1e-12
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(n_iter):
        Kv = K @ v
        Kv[Kv == 0] = 1e-12
        u = a / Kv
        KTu = K.T @ u
        KTu[KTu == 0] = 1e-12
        v = b / KTu
    return (u[:, None] * K) * v[None, :]


def cdist_sq(X, Y):
    X2 = np.sum(X**2, axis=1)[:, None]
    Y2 = np.sum(Y**2, axis=1)[None, :]
    return X2 + Y2 - 2 * (X @ Y.T)


def free_support_ot(X, w, m, rng, eps=1e-1, max_outer=100, sinkhorn_iter=1000):
    n, _ = X.shape
    idx0 = rng.choice(n, size=m, replace=True, p=w)
    Y = X[idx0] + 0.03 * rng.normal(size=(m, X.shape[1]))
    b = np.ones(m) / m
    for _ in range(max_outer):
        C = cdist_sq(X, Y)
        Pi = sinkhorn(w, b, C, eps=eps, n_iter=sinkhorn_iter)
        col_sums = Pi.sum(axis=0) + 1e-12
        Y = (Pi.T @ X) / col_sums[:, None]
    return Y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_PNG,
                    help="output png path")
    args = ap.parse_args()
    rng = np.random.default_rng(26)

    n, m = 100, 30
    mu1 = np.array([-2.0, 0.0])
    mu2 = np.array([2.5, 2.0])
    Sigma = np.array([[1.0, 0.4], [0.4, 1.2]])

    z = rng.uniform(size=n) < 0.5
    X = np.empty((n, 2))
    X[z] = rng.multivariate_normal(mu1, Sigma, size=z.sum())
    X[~z] = rng.multivariate_normal(mu2, Sigma, size=(~z).sum())

    like = 0.94 * gaussian_pdf(X, mu1, Sigma * 1.1) + 0.06 * gaussian_pdf(X, mu2, Sigma * 1.1)
    w = like / like.sum()

    Y_ot = free_support_ot(X, w, m, rng)
    idx_multi = rng.choice(n, size=m, replace=True, p=w)
    Y_multi = X[idx_multi].copy()

    grid_x, grid_y = np.meshgrid(
        np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 160),
        np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 160),
    )
    G = np.c_[grid_x.ravel(), grid_y.ravel()]
    p_mix = 0.5 * gaussian_pdf(G, mu1, Sigma) + 0.5 * gaussian_pdf(G, mu2, Sigma)
    p_mix = p_mix.reshape(grid_x.shape)
    x_min, x_max = grid_x.min() + 1, grid_x.max() - 1
    y_min, y_max = grid_y.min() + 1.7, grid_y.max() - 1.7

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.4), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.3})
    panels = [
        ("Weighted particles",
         lambda ax: ax.scatter(X[:, 0], X[:, 1], s=2000 * w, marker="o", alpha=0.45)),
        ("Multinomial resampling",
         lambda ax: ax.scatter(Y_multi[:, 0], Y_multi[:, 1], s=26, marker="+", alpha=0.85, color="orange")),
        ("Proposed resampling",
         lambda ax: ax.scatter(Y_ot[:, 0], Y_ot[:, 1], s=26, marker="x", alpha=0.95, color="red")),
    ]
    for ax, (title, draw) in zip(axes, panels):
        ax.contour(grid_x, grid_y, p_mix, levels=7, linewidths=0.7, alpha=0.7)
        draw(ax)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
