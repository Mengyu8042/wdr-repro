"""High-dimensional linear Gaussian spatial model of Section 5.2.

Model (linear spatial model):
    x_{t+1} = alpha x_t + eta_t,  eta_t ~ N(0, Q),   Q = q * Matern_{nu,rho}
    y_t     = H x_t + eps_t,      eps_t ~ N(0, r I)
on an n x n lattice, d = n^2. Q is dense, so drawing the spatially correlated
innovation costs one dense (m,d) x (d,d) GEMM per step — the propagation
expense is intrinsic to the field model.
"""
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


@dataclass
class FieldModel:
    alpha: float
    LQ: np.ndarray           # (d, d) lower Cholesky factor of Q
    Q: np.ndarray            # (d, d)
    obs_idx: np.ndarray      # (p,) observed site indices
    r: float
    n: int

    @property
    def d(self) -> int:
        return self.Q.shape[0]

    @property
    def p(self) -> int:
        return self.obs_idx.size


def matern_cov(n: int, nu: float, rho: float, q: float,
               jitter: float = 1e-8) -> np.ndarray:
    """Dense Matern covariance on an n x n unit-spacing lattice.
    nu in {0.5, 1.5, 2.5} uses the closed forms."""
    g = np.arange(n)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    s = dist / rho
    if abs(nu - 0.5) < 1e-9:
        C = np.exp(-s)
    elif abs(nu - 1.5) < 1e-9:
        a = np.sqrt(3.0) * s
        C = (1.0 + a) * np.exp(-a)
    elif abs(nu - 2.5) < 1e-9:
        a = np.sqrt(5.0) * s
        C = (1.0 + a + a * a / 3.0) * np.exp(-a)
    else:
        raise ValueError("nu must be 0.5, 1.5 or 2.5")
    C = q * C
    C[np.diag_indices_from(C)] += jitter * q
    return C


# Defaults follow the paper's Section 5.2 model (as reported:
# nu=1.5, rho=6, alpha=0.95, q=0.5, r=1 on a 45x45 lattice); obs_frac is
# always passed explicitly (paper uses 0.1).
def build_model(n: int = 45, nu: float = 1.5, rho: float = 6.0,
                alpha: float = 0.95, q_scale: float = 0.5,
                r_scale: float = 1.0, obs_frac: float = 1.0,
                seed: int = 7301) -> FieldModel:
    Q = matern_cov(n, nu, rho, q_scale)
    LQ = np.linalg.cholesky(Q)
    d = n * n
    if obs_frac >= 1.0:
        idx = np.arange(d)
    else:
        k = max(1, int(round(obs_frac * d)))
        idx = np.sort(np.random.default_rng(seed).choice(d, k, replace=False))
    return FieldModel(alpha=alpha, LQ=LQ, Q=Q, obs_idx=idx, r=r_scale, n=n)


def propagate(model: FieldModel, X: np.ndarray, rng: np.random.Generator,
              n_descendants: int = 1) -> np.ndarray:
    Xp = np.repeat(X, n_descendants, axis=0) if n_descendants > 1 else X
    z = rng.standard_normal((Xp.shape[0], model.d))
    return model.alpha * Xp + z @ model.LQ.T


def simulate(model: FieldModel, T: int, seed: int):
    rng = np.random.default_rng(seed)
    d = model.d
    x = (rng.standard_normal(d) @ model.LQ.T)
    X = np.empty((T, d))
    Y = np.empty((T, model.p))
    for t in range(T):
        x = model.alpha * x + rng.standard_normal(d) @ model.LQ.T
        X[t] = x
        Y[t] = x[model.obs_idx] + np.sqrt(model.r) * rng.standard_normal(model.p)
    return X, Y


def kf_mean_timevarying(model: FieldModel, Y: np.ndarray, device: str = "cuda"):
    """Full time-varying Kalman filter from the stationary prior. Used only to
    validate the steady-state gain; O(d^3) per step."""
    import torch
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    d, p = model.d, model.p
    Q = torch.tensor(model.Q, dtype=torch.float64, device=dev)
    idx = torch.tensor(model.obs_idx, dtype=torch.long, device=dev)
    R = model.r * torch.eye(p, dtype=torch.float64, device=dev)
    a = model.alpha
    Sig = Q / (1.0 - a * a)
    mu = torch.zeros(d, dtype=torch.float64, device=dev)
    out = np.empty((Y.shape[0], d))
    for t in range(Y.shape[0]):
        mu_p = a * mu
        P = a * a * Sig + Q
        PH = P[:, idx]
        S = PH[idx, :] + R
        K = torch.linalg.solve(S, PH.T).T
        yv = torch.tensor(Y[t], dtype=torch.float64, device=dev)
        mu = mu_p + K @ (yv - mu_p[idx])
        Sig = P - K @ PH.T
        out[t] = mu.cpu().numpy()
    return out


def steady_state_covariance(model: FieldModel, device: str = "cuda",
                            tol: float = 1e-10, max_iter: int = 5000):
    """Iterate the Riccati recursion of the Kalman filter to its steady-state
    filtering covariance, in float64 on the GPU when available. The recursion
    does not involve the data, so one matrix serves every replication, method
    and particle size. Returns (Sigma, iterations, final relative change)."""
    import torch
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    d, p = model.d, model.p
    Q = torch.tensor(model.Q, dtype=torch.float64, device=dev)
    idx = torch.tensor(model.obs_idx, dtype=torch.long, device=dev)
    a2 = model.alpha ** 2
    R = model.r * torch.eye(p, dtype=torch.float64, device=dev)
    Sig = Q.clone()
    for it in range(max_iter):
        P = a2 * Sig + Q                       # predictive covariance
        PH = P[:, idx]                         # P H^T
        S = PH[idx, :] + R                     # H P H^T + R
        K = torch.linalg.solve(S, PH.T).T      # P H^T S^{-1}
        Sig_new = P - K @ PH.T
        delta = torch.max(torch.abs(Sig_new - Sig)) / max(
            float(torch.max(torch.abs(Sig_new))), 1e-300)
        Sig = Sig_new
        if float(delta) < tol:
            break
    return Sig.cpu().numpy(), it + 1, float(delta)


class EnergyAccumulator:
    """Per-step energy score averaged over the steady window, and, when 
    `gauss_Sigma` is given, the blockwise energy distance of the particles to 
    the exact filtering distribution N(mu_t, Sigma) (`gauss_ed.py`).

    Sigma is the steady-state filtering covariance: the covariance recursion
    of the model does not involve the data, so one matrix serves every
    replication, method and particle size, and it has converged well before
    `t_start`. For the block particle filter, the energy distance is evaluated 
    block by block against the block marginal of N(mu_t, Sigma) and averaged 
    over the blocks."""

    def __init__(self, refs, t_start, device="cuda", gauss_mu=None,
                 gauss_Sigma=None, blocks=None, pre_stride=2):
        import torch
        self.torch = torch
        use = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.dev = torch.device(use)
        self.refs = {k: torch.as_tensor(np.ascontiguousarray(v),
                                        dtype=torch.float64, device=self.dev)
                     for k, v in refs.items()}
        self.t_start = int(t_start)
        self.first = {k: [] for k in refs}
        self.spread = []
        self.ed_on = gauss_Sigma is not None
        self.pre_stride = int(pre_stride)
        self.ed_pre = []
        if self.ed_on:
            from gauss_ed import GaussianEnergyTerms
            S = np.asarray(gauss_Sigma)
            self.gmu = torch.as_tensor(np.ascontiguousarray(gauss_mu),
                                       dtype=torch.float64, device=self.dev)
            self.blocks = [torch.as_tensor(np.asarray(b), dtype=torch.long,
                                           device=self.dev) for b in blocks]
            self.bterm = [GaussianEnergyTerms(S[np.ix_(np.asarray(b),
                                                       np.asarray(b))],
                                              device=self.dev)
                          for b in blocks]

    def step_pre(self, t, Xf, W):
        """Blockwise energy distance of the propagated particles `Xf` under
        the block weights `W` (blocks by particles), at every `pre_stride`-th
        step of the steady window."""
        if not self.ed_on or t < self.t_start or (t - self.t_start) % self.pre_stride:
            return
        torch = self.torch
        from gauss_ed import energy_distance
        xf = torch.as_tensor(Xf, dtype=torch.float64, device=self.dev)
        Wt = torch.as_tensor(W, dtype=torch.float64, device=self.dev)
        vals = []
        for k, (b, term) in enumerate(zip(self.blocks, self.bterm)):
            wt = Wt[k] / Wt[k].sum()
            vals.append(energy_distance(xf[:, b], wt, self.gmu[t][b], term))
        vals = np.asarray(vals, dtype=float)
        # NaN marks a block whose weights sit on one particle (distance undefined)
        self.ed_pre.append(float(np.nanmean(vals)) if np.isfinite(vals).any()
                           else float("nan"))

    def _step_on(self, t, X):
        torch = self.torch
        xt = torch.as_tensor(X, dtype=torch.float64, device=self.dev)
        self.spread.append(float(torch.cdist(xt, xt, p=2.0).mean()))
        for k, R in self.refs.items():
            self.first[k].append(float(torch.linalg.vector_norm(
                xt - R[t][None, :], dim=1).mean()))

    def step(self, t, X):
        if t < self.t_start:
            return
        try:
            self._step_on(t, X)
        except RuntimeError:
            if self.dev.type == "cpu":
                raise
            self.dev = self.torch.device("cpu")
            self.refs = {k: v.cpu() for k, v in self.refs.items()}
            self._step_on(t, X)

    def summary(self):
        sp = np.asarray(self.spread)
        out = {f"es_{k}": float((np.asarray(v) - 0.5 * sp).mean())
               for k, v in self.first.items()}
        if self.ed_on:
            out["ed"] = float(np.nanmean(self.ed_pre))
        return out


def metrics(est, oracle_mean, T):
    steady = slice(T // 2, T)
    def rmse(a, b):
        return float(np.sqrt(np.mean((a[steady] - b[steady]) ** 2)))
    return dict(rmse_oracle=rmse(est, oracle_mean))


# ---------------------------------------------------------------------------
# Localized (block) particle filter
#
# Rebeschini & van Handel (2015, Ann. Appl. Probab.) block particle filter:
# partition the lattice into blocks; each block carries its own weights,
# computed from the observations inside that block's localization
# neighbourhood, and is resampled independently. Any resampling scheme can be
# substituted blockwise, which makes the resampler comparison possible at a 
# state dimension where a global bootstrap filter has collapsed.
# ---------------------------------------------------------------------------

def make_blocks(n: int, block: int):
    """Partition an n x n lattice into square blocks of side `block`.
    Returns a list of flat index arrays."""
    out = []
    for i0 in range(0, n, block):
        for j0 in range(0, n, block):
            ii, jj = np.meshgrid(np.arange(i0, min(i0 + block, n)),
                                 np.arange(j0, min(j0 + block, n)),
                                 indexing="ij")
            out.append((ii.ravel() * n + jj.ravel()).astype(np.int64))
    return out


def block_obs_sets(model: FieldModel, blocks, radius: float):
    """For each block, the positions (into the observation vector) of the
    observations within `radius` lattice units of the block's sites."""
    n = model.n
    obs_sites = model.obs_idx
    oy, ox = obs_sites // n, obs_sites % n
    sets = []
    for b in blocks:
        by, bx = b // n, b % n
        dy = np.abs(oy[None, :] - by[:, None])
        dx = np.abs(ox[None, :] - bx[:, None])
        near = (np.sqrt(dy ** 2 + dx ** 2) <= radius).any(axis=0)
        sets.append(np.flatnonzero(near))
    return sets


def block_filter(model: FieldModel, Y: np.ndarray, m: int, seed: int,
                 block: int = 5, radius: float = None,
                 R_desc: int = 3,
                 resampler: Optional[Callable] = None,
                 batched_resampler: Optional[Callable] = None,
                 es_accum: Optional["EnergyAccumulator"] = None):
    """Block particle filter with a substitutable blockwise resampler.

    `batched_resampler`, if given, receives all blocks at once as
    (B, n_src, block_dim) particles and (B, n_src) weights and returns
    (B, n_out, block_dim); it requires equal-sized blocks."""
    if radius is None:
        radius = block
    rng = np.random.default_rng(seed)
    T, d = Y.shape[0], model.d
    inv2r = 0.5 / model.r
    blocks = make_blocks(model.n, block)
    osets = block_obs_sets(model, blocks, radius)
    X = rng.standard_normal((m, d)) @ model.LQ.T
    est = np.empty((T, d))      # ensemble mean
    blk_dim = float(np.mean([b.size for b in blocks]))
    for t in range(T):
        Xf = propagate(model, X, rng, R_desc)
        resid = Y[t][None, :] - Xf[:, model.obs_idx]
        sq = resid ** 2
        Xn = np.empty((m, d))
        W = np.empty((len(blocks), Xf.shape[0]))
        for k, o in enumerate(osets):
            lw = -inv2r * sq[:, o].sum(axis=1)
            lw -= lw.max()
            w = np.exp(lw)
            W[k] = w / w.sum()
        if es_accum is not None:
            es_accum.step_pre(t, Xf, W)
        if batched_resampler is not None:
            sizes = {b.size for b in blocks}
            if len(sizes) != 1:
                raise ValueError("batched resampler requires equal block sizes; "
                                 f"got {sorted(sizes)}")
            Xb = np.stack([Xf[:, b] for b in blocks], axis=0)
            out = batched_resampler(Xb, W, m)
            for k, b in enumerate(blocks):
                Xn[:, b] = out[k]
        else:
            for k, b in enumerate(blocks):
                Xn[:, b] = resampler(Xf[:, b], W[k], rng, m)
        X = Xn
        est[t] = Xn.mean(axis=0)
        if es_accum is not None:
            es_accum.step(t, X)
    return {"est": est, "block_dim": blk_dim}
