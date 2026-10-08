# -*- coding: utf-8 -*-
"""OT resampling of VAE latent features for diffusion image experiments.

Called by wdr_latent and t_wdr_latent in smc_utils.py. Inputs are a
PyTorch tensor of latent features (N, d) and NumPy weights (N,).
resample_wdr returns equal-weight features;
resample_t_wdr also returns updated weights.
"""
import numpy as np
import ot
from dataclasses import dataclass
import torch

@dataclass
class OTConfig:
    max_outer_iter: int = 10
    outer_tol: float = 1e-4
    
@torch.no_grad()
def sqeuclidean_cost(x, y):
    # x: (m,d), y: (n,d)
    # ‖x‖^2 + ‖y‖^2 - 2 x y^T
    x2 = (x * x).sum(dim=1, keepdim=True)          # (m,1)
    y2 = (y * y).sum(dim=1, keepdim=False).unsqueeze(0)  # (1,n)
    return x2 + y2 - 2.0 * (x @ y.t())


@torch.no_grad()
def update_ot_coupling(x, y, a, b):
    cost_matrix = sqeuclidean_cost(x, y)  # squared Euclidean cost
    cost_matrix = torch.nan_to_num(cost_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    max_cost = cost_matrix.max()
    if torch.isnan(max_cost) or max_cost <= 0:
        cost_matrix.zero_()
    else:
        cost_matrix.div_(max_cost)
    P = ot.emd(a.detach().cpu().numpy(),
               b.detach().cpu().numpy(),
               cost_matrix.detach().cpu().numpy())
    return torch.as_tensor(P, dtype=torch.float32, device=a.device)


@torch.no_grad()
def block_coordinate_descent_ot(a, b, y, rng, config: OTConfig, x_init=None):
    device = y.device
    a_t = torch.as_tensor(a, dtype=torch.float32, device=device)
    b_t = torch.as_tensor(b, dtype=torch.float32, device=device)
    y_t = y.to(dtype=torch.float32)

    m = a_t.numel()
    n, d = y_t.shape

    if x_init is None:
        x_t = torch.as_tensor(rng.random((m, d)), dtype=torch.float32, device=device)
    else:
        x_t = x_init.to(dtype=torch.float32, device=device)

    for _ in range(config.max_outer_iter):
        x_old = x_t.clone()
        # -- (1) update P with x fixed
        P_t = update_ot_coupling(x_t, y_t, a_t, b_t)
        # -- (2) update x with P fixed
        x_t = (P_t @ y_t) / a_t.unsqueeze(1)
        rel = torch.linalg.norm(x_t - x_old) / (torch.linalg.norm(x_old) + 1e-16)
        if rel.item() < float(config.outer_tol):
            break

    return x_t, P_t


def resample_wdr(particles, weights, rng, new_size, config: OTConfig = OTConfig()):
    device = particles.device
    a = np.ones(new_size) / new_size
    b = weights
    N = len(weights)

    # initialize x by random sampling from the input particles
    int_indices = rng.choice(np.arange(N), size=new_size, p=weights)
    int_indices_t = torch.as_tensor(np.ascontiguousarray(int_indices), dtype=torch.long, device=device)
    x_init = particles[int_indices_t]

    # alternating BCD iterations
    _, P_t = block_coordinate_descent_ot(a, b, particles, rng, config, x_init)
    P_np = P_t.detach().cpu().numpy()
    n = P_np.shape[1]
    idx = np.array([rng.choice(n, p=row / row.sum()) for row in P_np])
    idx_t = torch.as_tensor(np.ascontiguousarray(idx), dtype=torch.long, device=device)
    return particles[idx_t], idx



# ------------- Truncated WDR -------------
def _sanitize_prob_np(w: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if not np.isfinite(s) or s <= eps:
        raise ValueError("weights are all zero or invalid.")
    return w / s


def _solve_c_and_split_indices(weights: np.ndarray, new_size: int, tol: float = 1e-14):
    q = _sanitize_prob_np(weights)
    M = q.size
    N = int(new_size)

    # Sort descending. .copy() makes the result contiguous (positive stride),
    # avoiding negative-stride downstream slices that torch.as_tensor rejects.
    order = np.argsort(q)[::-1].copy()
    q_sorted = q[order]
    cumsum = np.concatenate([[0.0], np.cumsum(q_sorted)])

    # Try L = number of always-kept particles
    L_max = min(N, M)
    c_found = None
    L_found = None

    for L in range(0, L_max + 1):
        tail_mass = 1.0 - cumsum[L] 
        if N == L:
            # then we must have tail_mass == 0 to satisfy N = L + c * tail_mass
            if tail_mass <= tol:
                c_found = np.inf
                L_found = L
                break
            else:
                continue

        if tail_mass <= tol:
            # Not enough mass left to allocate N-L points
            continue

        c = (N - L) / tail_mass
        thr = 1.0 / c

        cond1 = (L == 0) or (q_sorted[L - 1] >= thr - 10 * tol)
        cond2 = (L == M) or (q_sorted[L] <  thr + 10 * tol)

        if cond1 and cond2:
            c_found = c
            L_found = L
            break

    if c_found is None:
        # Fallback: bisection on c (monotone) to be safe numerically
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
        # threshold = 0 => set2 should have (almost) zero mass; choose set1 as top N
        set1_idx = order[:N]
        set2_idx = order[N:]
        return c, set1_idx, set2_idx

    thr = 1.0 / c
    set1_mask = (q >= thr)
    set1_idx = np.where(set1_mask)[0]
    set2_idx = np.where(~set1_mask)[0]

    # Guard: due to floating ties, ensure L <= N
    # If L > N (rare tie issue), keep only top-N as set1, rest to set2
    if set1_idx.size > N:
        original_set1 = set1_idx
        top = np.argsort(q[original_set1])[::-1]
        set1_idx = original_set1[top[:N]]
        # push the rest into set2
        rest = original_set1[top[N:]]
        set2_idx = np.concatenate([set2_idx, rest], axis=0)

    return c, set1_idx, set2_idx


def resample_t_wdr(
    particles,
    weights: np.ndarray,
    rng: np.random.Generator,
    new_size: int,
    config: OTConfig = OTConfig(),
):
    device = particles.device
    q = _sanitize_prob_np(weights)
    N, d = particles.shape

    c, set1_idx, set2_idx = _solve_c_and_split_indices(q, new_size)
    L = set1_idx.size
    m = new_size - L

    # ---- Degenerate case: c=inf should imply "no set2 mass needed" ----
    if np.isinf(c) or m <= 0:
        idx = set1_idx[:new_size]
        idx_t = torch.as_tensor(np.ascontiguousarray(idx), dtype=torch.long, device=device)
        X_new = particles[idx_t]
        w_new = _sanitize_prob_np(q[idx])

        perm = rng.permutation(new_size)
        perm_t = torch.as_tensor(np.ascontiguousarray(perm), dtype=torch.long, device=device)
        X_new = X_new[perm_t]
        w_new = w_new[perm]
        idx = idx[perm]
        return X_new, w_new, idx

    # --- set1: keep as-is ---
    set1_idx_t = torch.as_tensor(np.ascontiguousarray(set1_idx), dtype=torch.long, device=device)
    part1 = particles[set1_idx_t]
    w1 = q[set1_idx]

    # --- set2: WDR ---
    set2_idx_t = torch.as_tensor(np.ascontiguousarray(set2_idx), dtype=torch.long, device=device)
    part2_src = particles[set2_idx_t]
    w2_src = q[set2_idx]
    w2_mass = float(w2_src.sum())

    if w2_mass < 1e-12 and set1_idx.size > 0:
        part2_src = part1
        w2_norm = w1 / (w1.sum() + 1e-16)
        set2_idx = set1_idx
    else:
        if w2_mass < 1e-12:
            w2_norm = np.full_like(w2_src, 1.0 / max(w2_src.size, 1))
        else:
            w2_norm = w2_src / w2_mass

    # call WDR on set2
    part2, set2_resamp_idx = resample_wdr(
        particles=part2_src,
        weights=w2_norm,
        rng=rng,
        new_size=m,
        config=config,
    )
    w2 = np.full(m, 1.0 / c, dtype=np.float64)

    new_particles = torch.cat([part1, part2], dim=0)
    new_weights = np.concatenate([w1, w2], axis=0)
    new_idxes = np.concatenate([set1_idx, set2_idx[set2_resamp_idx]], axis=0)

    new_weights = _sanitize_prob_np(new_weights)

    perm = rng.permutation(new_size)
    perm_t = torch.as_tensor(np.ascontiguousarray(perm), dtype=torch.long, device=device)
    new_particles = new_particles[perm_t]
    new_weights = new_weights[perm]
    new_idxes = new_idxes[perm]

    return new_particles, new_weights, new_idxes



