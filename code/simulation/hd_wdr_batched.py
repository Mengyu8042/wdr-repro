"""Batched blockwise WDR for the block particle filter of the spatial model.

The canonical `wasserstein_dirichlet_resampling` solves one OT problem per
call and is dominated by CUDA launch overhead. A block particle filter needs 
one call per block -- 81 blocks at d=2025 -- which is not runnable.

The blocks are independent problems of identical shape, so this module runs
them as one batch: every canonical operation is reproduced with a leading
batch dimension.
"""
import numpy as np
import torch


def _sq_cost(x, y):
    """(B,m,d), (B,n,d) -> (B,m,n) squared Euclidean cost."""
    x2 = (x * x).sum(-1).unsqueeze(2)
    y2 = (y * y).sum(-1).unsqueeze(1)
    return (x2 + y2 - 2.0 * torch.bmm(x, y.transpose(1, 2))).clamp_min(0.0)


def _ipot_batched(a, b, M, reg, n_outer, inner_L, stop_thr):
    """Batched IPOT. a:(B,m), b:(B,n), M:(B,m,n) -> P:(B,m,n)."""
    B, m, n = M.shape
    K = torch.exp(-M / reg)
    Q = K / (m * n)
    a_col = a.unsqueeze(2)
    b_col = b.unsqueeze(2)
    a_scale = a_col.abs().clamp_min(1e-16)
    v = torch.ones((B, n, 1), dtype=M.dtype, device=M.device)
    u = torch.ones((B, m, 1), dtype=M.dtype, device=M.device)
    for it in range(n_outer):
        for _ in range(inner_L):
            u = a_col / (torch.bmm(Q, v) + 1e-16)
            v = b_col / (torch.bmm(Q.transpose(1, 2), u) + 1e-16)
        if it % 10 == 1:
            res = ((u * torch.bmm(Q, v) - a_col).abs() / a_scale).amax()
            if float(res) < stop_thr:
                break
        Q = Q * K * u * v.transpose(1, 2)
    return u * Q * v.transpose(1, 2)


def wdr_blocks(Xsrc, w, n_out, gen, cfg, tau=30.0, sparsify=None,
               spar_ratio=0.3):
    """WDR for a batch of equally sized spatial blocks.

    Xsrc: PyTorch tensor of shape (B, n_src, d), one particle set per block.
    w: PyTorch tensor of shape (B, n_src), with nonnegative weights and
       positive total weight in each block.
    n_out: number of output particles per block.
    gen: PyTorch generator used for source subsampling and support initialization.
    cfg: OT solver configuration; see CANON_CFG.

    Returns equal-weight particles of shape (B, n_out, d).
    Dirichlet draws use PyTorch's global generator; seed it as well as gen.
    """
    B, n_src, d = Xsrc.shape
    dev, dt = Xsrc.device, Xsrc.dtype
    use_spar = (max(n_src, n_out) > 600) if sparsify is None else bool(sparsify)

    if use_spar:
        k = max(2, int(spar_ratio * n_src))
        probs = torch.sqrt(w.clamp_min(0.0))
        probs = probs / probs.sum(1, keepdim=True)
        idx = torch.multinomial(probs, k, replacement=False, generator=gen)
        Y = torch.gather(Xsrc, 1, idx.unsqueeze(-1).expand(B, k, d))
        b = torch.gather(w, 1, idx)
        b = b / b.sum(1, keepdim=True)
    else:
        Y, b = Xsrc, w / w.sum(1, keepdim=True)

    a = torch.full((B, n_out), 1.0 / n_out, dtype=dt, device=dev)
    start = torch.multinomial(b, n_out, replacement=True, generator=gen)
    x = torch.gather(Y, 1, start.unsqueeze(-1).expand(B, n_out, d)).clone()

    P = None
    for _ in range(cfg["max_outer_iter"]):
        x_old = x
        C = _sq_cost(x, Y)
        cmax = C.amax(dim=(1, 2), keepdim=True).clamp_min(1e-30)
        C = C / cmax
        P = _ipot_batched(a, b, C, cfg["reg"], cfg["max_inner_iter"],
                          cfg["ipot_inner_iter"], cfg["inner_tol"])
        x = torch.bmm(P, Y) / a.unsqueeze(2)
        rel = (torch.linalg.norm((x - x_old).reshape(B, -1), dim=1)
               / (torch.linalg.norm(x_old.reshape(B, -1), dim=1) + 1e-16))
        if float(rel.max()) < cfg["outer_tol"]:
            break

    if tau <= 0:
        return x
    if not np.isfinite(tau):
        return x            # canonical OT(det) branch: deterministic barycentre
    conc = (tau * n_out * P).clamp_min(1e-8)
    g = torch._standard_gamma(conc)                    # (B, n_out, n_tgt)
    alpha = g / g.sum(-1, keepdim=True).clamp_min(1e-30)
    return torch.bmm(alpha, Y)


CANON_CFG = dict(max_outer_iter=10, max_inner_iter=50, ipot_inner_iter=4,
                 reg=2e-2, outer_tol=1e-4, inner_tol=1e-2)
