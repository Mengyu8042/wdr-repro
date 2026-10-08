"""Energy distance between a weighted particle set and a Gaussian distribution.
"""
import numpy as np

try:
    import torch
except ImportError:                                     # numpy-only callers
    torch = None

QUAD_S = 40.0        # half-width of the s = log t grid
QUAD_N = 200         # grid points; 200 already gives ~1e-7 relative error


def _trapz_uniform(y, s):
    """Trapezoid rule on a uniform grid, last axis."""
    ds = float(s[1] - s[0])
    return (y.sum(-1) - 0.5 * (y[..., 0] + y[..., -1])) * ds


class GaussianEnergyTerms:
    """The Gaussian terms of the energy distance to N(mu, Sigma).

    Construct once per Sigma, then call cross() per timestep.  Everything is
    float64, matching the metric convention used elsewhere in this code.
    """

    def __init__(self, Sig, device=None, S=QUAD_S, nq=QUAD_N):
        self.torch = torch is not None and device is not None
        if self.torch:
            dev = torch.device(device)
            Sg = torch.as_tensor(np.ascontiguousarray(Sig),
                                 dtype=torch.float64, device=dev)
            lam, U = torch.linalg.eigh(Sg)
            lam = torch.clamp(lam, min=1e-300)
            self.s = torch.linspace(-S, S, nq, dtype=torch.float64, device=dev)
            self.dev = dev
        else:
            lam, U = np.linalg.eigh(np.asarray(Sig, dtype=np.float64))
            lam = np.clip(lam, 1e-300, None)
            self.s = np.linspace(-S, S, nq)
        self.U, self.lam = U, lam
        # E||Y - Y'||: Y - Y' ~ N(0, 2 Sigma) since Y, Y' are iid Gaussian, so
        # the same formula with nu = 0 and eigenvalues doubled.
        zero = (torch.zeros((1, lam.shape[0]), dtype=torch.float64,
                            device=self.dev) if self.torch
                else np.zeros((1, lam.shape[0])))
        self.self_term = float(self._e_norm(zero, 2.0 * lam)[0])

    def _e_norm(self, M, lam):
        """E||N(m_k, Sigma)|| for each row m_k of M, M already expressed in the
        eigenbasis of Sigma.  Scale-normalised for conditioning."""
        xp = torch if self.torch else np
        c = float(lam.mean() ** 0.5)
        lam_n, M_n = lam / c ** 2, M / c
        s = self.s
        t = xp.exp(s)
        a = 1.0 + 2.0 * (t[:, None] * lam_n[None, :])          # (nq, d)
        logdet = -0.5 * xp.log(a).sum(1)                       # (nq,)
        # quad[k, i] = sum_j M[k,j]^2 / a[i,j], as one matmul: (K,d) @ (d,nq)
        quad = (M_n ** 2) @ (1.0 / a).T
        phi = xp.exp(logdet[None, :] - t[None, :] * quad)      # (K, nq)
        integ = (1.0 - phi) * xp.exp(-0.5 * s)[None, :]
        return c * _trapz_uniform(integ, s) / (2.0 * np.sqrt(np.pi))

    def cross(self, X, mu):
        """E||x_i - Y|| for every particle, Y ~ N(mu, Sigma). Returns (N,)."""
        return self._e_norm((X - mu) @ self.U, self.lam)


def energy_distance(X, w, mu, terms):
    """ED( sum_i w_i delta_{x_i}, N(mu, Sigma) ) with the bias-corrected
    pairwise term.

    X     : (N, d) particles           w : (N,) weights, summing to 1
    mu    : (d,) filtering mean        terms : GaussianEnergyTerms for Sigma

    Returns NaN when the weights sit on a single particle (sum_i w_i^2 = 1),
    where the bias-corrected pairwise term is undefined.
    """
    wDw = _pairwise(X, w)
    den = 1.0 - float(w @ w)
    if den <= 1e-12:
        return float("nan")
    return (2.0 * float(w @ terms.cross(X, mu))
            - float(wDw) / den - terms.self_term)


def _pairwise(X, w):
    if torch is not None and torch.is_tensor(X):
        return w @ (torch.cdist(X, X) @ w)
    from scipy.spatial.distance import cdist
    return w @ cdist(X, X) @ w


def build_terms_sequence(Sigma_list, device=None, rtol=1e-12):
    """One GaussianEnergyTerms per timestep, sharing objects across timesteps
    whose covariance has stopped changing.
    """
    terms, prev, cur = [], None, None
    for Sig in Sigma_list:
        S = np.asarray(Sig, dtype=np.float64)
        if prev is None or not np.allclose(S, prev, rtol=rtol, atol=0.0):
            cur = GaussianEnergyTerms(S, device=device)
            prev = S
        terms.append(cur)
    n_unique = len({id(t) for t in terms})
    return terms, n_unique
