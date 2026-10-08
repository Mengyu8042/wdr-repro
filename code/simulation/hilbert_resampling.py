"""Hilbert-curve stratified resampling.

Reference: Li, Wang, Deng, and Liu (2022), "Stratification and optimal
resampling for sequential Monte Carlo," Biometrika.
"""

import numpy as np
from typing import Optional
from hilbertcurve.hilbertcurve import HilbertCurve


def _choose_p_auto(n_points: int, dim: int) -> int:
    """Choose the Hilbert curve order p so 2^(p*dim) >= 4*n_points."""
    if n_points <= 1:
        return 1
    target = int(np.ceil(np.log2(4 * n_points)))
    p = max(1, int(np.ceil(target / dim)))
    return p


def _to_uint_grid(points01: np.ndarray, p: int) -> np.ndarray:
    """Map points in [0,1]^d to the integer grid [0, 2^p - 1]^d."""
    base = 2 ** p
    return np.clip((points01 * base).astype(int), 0, base - 1)


def _normalize_to_unit_cube(X: np.ndarray, pad: float = 1e-9) -> np.ndarray:
    """Min-max scale each dimension to [0,1]; degenerate dims are set to 0.5."""
    X = np.asarray(X, dtype=float)
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    span = xmax - xmin
    out = np.empty_like(X)
    mask = span > 0
    out[:, mask] = (X[:, mask] - xmin[mask]) / (span[mask] + pad)
    out[:, ~mask] = 0.5
    return np.clip(out, 0.0, 1.0)


def resample_hilbert(
    particles: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    new_size: int,
    p: Optional[int] = None,
    pad: float = 1e-9,
    return_indices: bool = False,
):
    """Hilbert-curve stratified resampling.

    The output has equal weights and supports drawn from the input particle set.

    Args
        particles: array of shape (N, d).
        weights: array of shape (N,), need not be normalized.
        rng: np.random.Generator.
        new_size: number of resampled particles.
        p: Hilbert order; None to choose automatically.
        pad: small constant added to span to avoid division by zero.
        return_indices: if True also return the ancestor indices.

    Returns
        out: (new_size, d) resampled particles.
        idx_sel (optional): (new_size,) ancestor indices into the input order.
    """
    X = np.asarray(particles, dtype=float)
    w = np.asarray(weights, dtype=float)
    N, d = X.shape

    # Normalize weights
    w = np.maximum(w, 0.0)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        w = np.ones(N, dtype=float) / N
    else:
        w /= s

    # Scale particles to [0,1]^d
    X01 = _normalize_to_unit_cube(X, pad=pad)

    # Choose Hilbert order if not given
    if p is None:
        p = _choose_p_auto(N, d)

    # Map to the integer grid and compute Hilbert distances in batch
    grid_pts = _to_uint_grid(X01, p)
    HC = HilbertCurve(p, d)
    h_idx = np.array(HC.distances_from_points(grid_pts))

    # Sort particles by Hilbert key (stable sort)
    order = np.argsort(h_idx, kind="mergesort")
    w_sorted = w[order]

    # Stratified resampling: one set of positions, one searchsorted
    cdf = np.cumsum(w_sorted)
    cdf[-1] = 1.0  # numerical guard
    u0 = rng.random()
    positions = (u0 + np.arange(new_size)) / new_size
    idx_in_sorted = np.searchsorted(cdf, positions, side="left")

    # Map back to the original index space
    idx_sel = order[idx_in_sorted]
    out = X[idx_sel]

    if return_indices:
        return out, idx_sel
    return out
