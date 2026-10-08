"""Resampling methods for the state-space simulations.

Called by run_lg_grid.py, run_lg_stability.py, lg_cost_model.py,
and (for classical baselines) run_hd_filter.py. Inputs are NumPy
particles (N, d) and weights (N,). WDR returns equal-weight particles;
T-WDR also returns their weights.

Functions:
    resample_multinomial, resample_stratified, resample_residual
        classical resampling baselines.
    wasserstein_dirichlet_resampling
        WDR. Free-support OT via block coordinate descent followed by
        Dirichlet mixing of the empirical coupling.
    truncated_wasserstein_dirichlet_resampling
        T-WDR. Splits particles and applies WDR to the residual set.
"""
from dataclasses import dataclass, replace
from typing import Optional
import numpy as np
import ot
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def resample_multinomial(particles, weights, rng, new_size):
    """Multinomial resampling."""
    N = len(weights)
    indices = rng.choice(np.arange(N), size=new_size, p=weights)
    return particles[indices]


def resample_stratified(particles, weights, rng, new_size):
    """Stratified resampling."""
    cdf = np.cumsum(weights)
    positions = (rng.random(new_size) + np.arange(new_size)) / new_size
    idx = np.searchsorted(cdf, positions)
    return particles[idx]


def resample_residual(particles, weights, rng, new_size):
    """Multinomial residual resampling."""
    N = len(weights)
    # Expected copy count for each particle
    expected_counts = new_size * weights
    # Integer part: deterministic copy count
    deterministic_counts = np.floor(expected_counts).astype(int)
    N_det = np.sum(deterministic_counts)
    N_res = new_size - N_det

    new_particles = []
    # Deterministic allocation: replicate particles
    for i in range(N):
        if deterministic_counts[i] > 0:
            new_particles.extend([particles[i]] * deterministic_counts[i])

    # Random allocation: sample from residual weights
    if N_res > 0:
        residual_weights = expected_counts - deterministic_counts
        residual_weights /= residual_weights.sum()
        indices = rng.choice(np.arange(N), size=N_res, p=residual_weights)
        for idx in indices:
            new_particles.append(particles[idx])

    new_particles = np.array(new_particles)
    if new_particles.shape[0] != new_size:
        if new_particles.shape[0] > new_size:
            new_particles = new_particles[:new_size]
        else:
            additional = rng.choice(new_particles, size=new_size - new_particles.shape[0], replace=True)
            new_particles = np.concatenate([new_particles, additional], axis=0)
    return new_particles


@dataclass
class OTConfigNew:
    method: str = None
    max_outer_iter: int = 10
    max_inner_iter: int = 50
    outer_tol: float = 1e-4
    inner_tol: float = 1e-2
    reg: float = 2e-2
    ipot_inner_iter: int = 4
    spar_ratio: float = 0.3
    device: Optional[str] = None


def _device_or_default(device=None):
    return torch.device('cuda' if (device is None and torch.cuda.is_available()) else ('cpu' if device is None else device))


def _solve_emd(a, b, cost_matrix):
    """Exact OT via POT's EMD solver."""
    P_np = ot.emd(
        a.detach().cpu().numpy(),
        b.detach().cpu().numpy(),
        cost_matrix.detach().cpu().numpy(),
    )
    return torch.as_tensor(P_np, dtype=torch.float32, device=a.device)


@torch.no_grad()
def _sanitize_probabilities(probs: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Sanitize a probability tensor in place, with a fast path for clean inputs."""
    totals = probs.sum(dim=dim, keepdim=True)
    ok = torch.isfinite(totals) & (totals > 0)
    if bool(ok.all()):
        return probs / totals

    x = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    x = x.clamp_min_(0.0)
    totals = x.sum(dim=dim, keepdim=True)
    ok = torch.isfinite(totals) & (totals > 0)

    if x.ndim == 1:
        if not bool(ok):
            x.fill_(1.0)
            totals = x.sum(dim=dim, keepdim=True)
        return x / totals

    invalid = ~ok
    uniform = torch.ones_like(x)
    x = torch.where(invalid, uniform, x)
    totals = x.sum(dim=dim, keepdim=True)
    return x / totals


def _row_normalize(P: torch.Tensor) -> torch.Tensor:
    return _sanitize_probabilities(P, dim=1)


@torch.no_grad()
def sqeuclidean_cost(x, y):
    # x: (m,d), y: (n,d). Returns ||x_i - y_j||^2.
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=False).unsqueeze(0)
    return x2 + y2 - 2.0 * (x @ y.t())


@torch.no_grad()
def _sparipot(a, b, M, beta=1.0, L=1, numItermax=100, stopThr=1e-4, P0=None):
    """Inexact proximal OT (IPOT) solver."""
    m, n = M.shape
    K = torch.exp(-M / float(beta))

    if P0 is None:
        Q = K * (1.0 / (m * n))
    else:
        Q = K * P0

    a_col = a.view(m, 1)
    b_col = b.view(n, 1)
    a_scale = a_col.abs().clamp_min(1e-16)
    v = torch.ones((n, 1), dtype=M.dtype, device=M.device)
    executed_iters = 0

    for it in range(numItermax):
        executed_iters = it + 1
        for _ in range(L):
            denom_u = Q @ v
            u = a_col / (denom_u + 1e-16)
            denom_v = Q.transpose(0, 1) @ u
            v = b_col / (denom_v + 1e-16)

        if it % 10 == 1:
            row_rel_res = ((u * (Q @ v) - a_col).abs() / a_scale).amax()
            if float(row_rel_res) < stopThr:
                break

        Q.mul_(K)
        Q.mul_(u)
        Q.mul_(v.transpose(0, 1))

    P = u * Q * v.transpose(0, 1)
    return P


@torch.no_grad()
def update_ot_coupling(x, y, a, b, config, P0):
    cost_matrix = sqeuclidean_cost(x, y)
    cost_matrix = torch.nan_to_num(cost_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    max_cost = cost_matrix.max()
    if torch.isnan(max_cost) or max_cost <= 0:
        cost_matrix.zero_()
    else:
        cost_matrix.div_(max_cost)

    if config.method == "emd":
        P = _solve_emd(a, b, cost_matrix)

    elif config.method == "sparIPOT":
        try:
            P = _sparipot(
                a, b, cost_matrix,
                beta=config.reg,
                L=config.ipot_inner_iter,
                numItermax=config.max_inner_iter,
                stopThr=config.inner_tol,
                P0=P0,
            )
        except Exception as exc:
            print(f"[WARN] IPOT failed ({exc}); falling back to EMD.")
            P = _solve_emd(a, b, cost_matrix)

    else:
        raise ValueError(f"Unknown method: {config.method}")

    return P


@torch.no_grad()
def block_coordinate_descent_ot(a_t, b_t, y_t, config, x_init):
    """Free-support OT by alternating updates of (P, x).

    Solves
        min_{x, P} sum_{i,j} P_{ij} ||x_i - y_j||^2  s.t.  P 1 = a, P^T 1 = b.

    a, b are marginal masses, y is the fixed support of the source measure,
    and x is the free support of the target measure (initialized to x_init).
    Returns the final (x, P).
    """
    x_t = x_init.clone()
    P_t = None

    for _ in range(config.max_outer_iter):
        x_old = x_t.clone()
        P_t = update_ot_coupling(x_t, y_t, a_t, b_t, config, P_t)
        x_t = (P_t @ y_t) / a_t.unsqueeze(1)

        # Outer early-stop on relative change
        rel = torch.linalg.norm(x_t - x_old) / (torch.linalg.norm(x_old) + 1e-16)
        if rel.item() < float(config.outer_tol):
            break
    return x_t, P_t


def wasserstein_dirichlet_resampling(
    particles: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    new_size: int,
    config: OTConfigNew = OTConfigNew(),
    tau: float = 30.0,
):
    """WDR: Wasserstein-Dirichlet resampling.

    Pipeline
        1. Solve free-support OT via BCD to get coupling P (m x n) with row
           marginal 1/m and column marginal weights.
        2. For each row i set p^(i) = m * P[i] as the Dirichlet base probability.
        3. Sample alpha^(i) ~ Dirichlet(tau * p^(i)) and output
           z_i = sum_j alpha_j^(i) * x_j.

    Args
        particles: NumPy array of shape (N, d).
        weights: nonnegative NumPy array of shape (N,), with positive sum.
        rng: NumPy generator retained for the shared resampler interface;
            random draws in this function use PyTorch's global generator.
        new_size: number of output particles.
        config: OT solver settings and device.
        tau: Dirichlet concentration; tau=0 gives OT randomized resampling
            and tau=np.inf gives OT deterministic resampling.

    Returns
        Equal-weight new particles, shape (new_size, d).
    """
    device = _device_or_default(config.device)

    particles_t = torch.as_tensor(particles, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    weights_t = _sanitize_probabilities(weights_t, dim=0)

    N = int(particles_t.shape[0])

    if config.method is None:
        if new_size <= 200:
            config = replace(config, method="emd", device=str(device))
        else:
            config = replace(config, method="sparIPOT", device=str(device))
            prob = torch.sqrt(weights_t).clamp_min(0.0)
            prob = _sanitize_probabilities(prob, dim=0)
            s = int(config.spar_ratio * N)
            idx = torch.multinomial(prob, s, replacement=False)
            particles_t = particles_t[idx]
            weights_t = weights_t[idx]
            weights_t = _sanitize_probabilities(weights_t, dim=0)

    elif config.device is None:
        config = replace(config, device=str(device))

    a_t = torch.ones(new_size, dtype=torch.float32, device=device) / new_size
    b_t = weights_t

    # Initialize support
    int_indices = torch.multinomial(weights_t, new_size, replacement=True)
    x_init_t = particles_t[int_indices]

    # Block coordinate descent
    x_det_t, P_t = block_coordinate_descent_ot(
        a_t, b_t, particles_t, config, x_init_t
    )

    if tau == np.inf:
        out_t = x_det_t

    elif tau <= 0:
        # tau == 0: degenerates to OT randomized resampling
        probs = _row_normalize(P_t)
        idx = torch.multinomial(probs, 1, replacement=True).squeeze(-1)
        out_t = particles_t[idx]

    else:
        # Dirichlet mixing
        probs = _row_normalize(P_t)
        theta = tau * probs

        # Gamma sampling (only at nonzero positions to avoid degenerate parameters)
        mask = theta > 1e-20
        g = torch.zeros_like(theta)
        g[mask] = torch.distributions.Gamma(theta[mask], torch.ones_like(theta[mask])).sample()

        s = g.sum(dim=1, keepdim=True)
        w = g / s
        out_t = w @ particles_t

    return out_t.detach().cpu().numpy()


# ------------- Truncated WDR -------------
def _sanitize_prob_np(w: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if not np.isfinite(s) or s <= eps:
        raise ValueError("weights are all zero or invalid.")
    return w / s


def _solve_c_and_split_indices(weights: np.ndarray, new_size: int, tol: float = 1e-14):
    """Solve for c such that new_size = sum_j min(c * q_j, 1) and split into two index sets.

    set1 = {j : q_j >= 1/c}  -- kept deterministically.
    set2 = remaining indices -- to be resampled / transformed.
    """
    q = _sanitize_prob_np(weights)
    M = q.size
    N = int(new_size)
    order = np.argsort(q)[::-1]
    q_sorted = q[order]
    cumsum = np.concatenate([[0.0], np.cumsum(q_sorted)])
    L_max = min(N, M)
    c_found = None

    for L in range(0, L_max + 1):
        tail_mass = 1.0 - cumsum[L]
        if N == L:
            if tail_mass <= tol:
                c_found = np.inf
                break
            else:
                continue

        if tail_mass <= tol:
            continue

        c = (N - L) / tail_mass
        thr = 1.0 / c

        cond1 = (L == 0) or (q_sorted[L - 1] >= thr - 10 * tol)
        cond2 = (L == M) or (q_sorted[L] < thr + 10 * tol)

        if cond1 and cond2:
            c_found = c
            break

    if c_found is None:
        # Bisection fallback for numerical safety.
        def f(c):
            return np.minimum(c * q, 1.0).sum()

        lo, hi = 0.0, 1.0
        while f(hi) < N and hi < 1e18:
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(mid) < N:
                lo = mid
            else:
                hi = mid
        c_found = hi

    c = float(c_found)
    if np.isinf(c):
        set1_idx = order[:N]
        set2_idx = order[N:]
        return c, set1_idx, set2_idx

    thr = 1.0 / c
    set1_mask = (q >= thr)
    set1_idx = np.where(set1_mask)[0]
    set2_idx = np.where(~set1_mask)[0]

    if set1_idx.size > N:
        original_set1 = set1_idx
        top = np.argsort(q[original_set1])[::-1]
        set1_idx = original_set1[top[:N]]
        rest = original_set1[top[N:]]
        set2_idx = np.concatenate([set2_idx, rest], axis=0)

    return c, set1_idx, set2_idx


def truncated_wasserstein_dirichlet_resampling(
    particles: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    new_size: int,
    config: OTConfigNew = OTConfigNew(),
    tau: float = 30.0,
    shuffle: bool = True,
):
    """T-WDR: Truncated WDR.

    Pipeline
        1. Solve c such that new_size = sum_j min(c * q_j, 1).
        2. set1 = {j : q_j >= 1/c} kept with their original weights.
        3. set2 = remaining indices, run WDR to draw m = new_size - |set1|
           equal-weight particles, each assigned weight 1/c.

    The output has non-equal weights in general.

    Args
        particles: NumPy array of shape (N, d).
        weights: nonnegative NumPy array of shape (N,), with positive sum.
        rng: NumPy generator used to shuffle the output.
        new_size: number of output particles.
        config, tau: settings passed to WDR for the residual particle set.
        shuffle: whether to shuffle the output using rng.

    Returns
        New particles of shape (new_size, d) and normalized weights of
        shape (new_size,).

    WDR draws use PyTorch's global generator; seed it as well as rng.
    """
    q = _sanitize_prob_np(weights)
    c, set1_idx, set2_idx = _solve_c_and_split_indices(q, new_size)
    L = set1_idx.size
    m = new_size - L

    # Degenerate case: c == inf implies set2 carries no mass.
    if np.isinf(c) or m <= 0:
        idx = set1_idx
        if idx.size < new_size:
            idx = np.argsort(q)[::-1][:new_size]
        else:
            idx = idx[:new_size]
        X_new = particles[idx]
        w_new = _sanitize_prob_np(q[idx])
        if shuffle:
            perm = rng.permutation(new_size)
            X_new, w_new = X_new[perm], w_new[perm]
        return X_new, w_new

    # set1 kept as is
    part1 = particles[set1_idx]
    w1 = q[set1_idx]

    # set2 resampled via WDR
    part2_src = particles[set2_idx]
    w2_src = q[set2_idx]
    w2_mass = float(w2_src.sum())
    w2_norm = w2_src / w2_mass

    part2 = wasserstein_dirichlet_resampling(
        particles=part2_src,
        weights=w2_norm,
        rng=rng,
        new_size=m,
        config=config,
        tau=tau,
    )
    w2 = np.full(m, 1.0 / c, dtype=np.float64)

    new_particles = np.concatenate([part1, part2], axis=0)
    new_weights = np.concatenate([w1, w2], axis=0)
    new_weights = _sanitize_prob_np(new_weights)
    if shuffle:
        perm = rng.permutation(new_size)
        new_particles = new_particles[perm]
        new_weights = new_weights[perm]

    return new_particles, new_weights
