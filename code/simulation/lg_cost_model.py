"""Performance versus computing cost on the linear Gaussian model (Figure 4).

Fixes one n_substeps multiplier via --substep, then runs all particle counts
for the registered resamplers. This script is one experimental unit; a shell
wrapper launches it once per multiplier (see code/experiments/fig4_lg_cost.sh,
which drives it through run_lg_cost.py).
"""
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Performance-versus-cost study on the linear Gaussian model.")
    parser.add_argument("--Ns", dest="Ns", default="256,512,1024,2048",
                        type=str, help="Comma-separated list of particle counts")
    parser.add_argument("--R", dest="R", default=3, type=int, help="Number of descendants")
    parser.add_argument("--rep", dest="n_repeats", default=10, type=int, help="Number of replications")
    parser.add_argument("--T", dest="T", default=1000, type=int, help="Number of timesteps")
    parser.add_argument("--d", dest="d", default=50, type=int, help="State dimension")
    parser.add_argument("--alpha", dest="alpha", default=0.95, type=float,
                        help="AR(1) coefficient of the dynamics")
    parser.add_argument("--q-scale", dest="q_scale", default=0.5, type=float,
                        help="Process noise scale")
    parser.add_argument("--r-scale", dest="r_scale", default=1.0, type=float,
                        help="Observation noise scale")
    parser.add_argument("--kappa", dest="kappa", default=50, type=float,
                        help="Condition number of the process noise covariance")
    parser.add_argument("--cuda", dest="cuda_no", default="0", type=str,
                        help="CUDA device ids for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--parallel", dest="if_parallel", default=0, type=int,
                        help="Whether to run in parallel")
    parser.add_argument("--workers", dest="max_workers", default=8, type=int,
                        help="Max workers for parallel execution")
    parser.add_argument("--cv", dest="cv_threshold", default=2, type=float,
                        help="Weight-CV threshold that triggers resampling")
    parser.add_argument("--substep", dest="substep_multiplier", default=1, type=int,
                        help="Single sub-step multiplier for this run")
    return parser.parse_args()

args = parse_args()
Ns = [int(x) for x in args.Ns.split(",")]
R = args.R
n_repeats = args.n_repeats
T_steps = args.T
d = args.d
ALPHA = args.alpha
q_scale = args.q_scale
r_scale = args.r_scale
kappa = args.kappa
cuda_no = args.cuda_no
if_parallel = args.if_parallel
MAX_WORKERS = args.max_workers
CV_THRES = args.cv_threshold
SUBSTEP_MULTIPLIER = args.substep_multiplier

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_no

import numpy as np
import time, json, pickle
from timeit import default_timer as timer
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp
import torch

from hilbert_resampling import resample_hilbert
from lg_common import (
    build_lg_model, kalman_filter, simulate_lg_data,
)
from ot_resampling import (
    resample_multinomial, resample_residual, resample_stratified,
    wasserstein_dirichlet_resampling, OTConfigNew,
)


def _resample_wdr(particles, weights, rng, new_size):
    cfg = OTConfigNew(max_inner_iter=max(10, int(new_size / 40)))
    return wasserstein_dirichlet_resampling(particles, weights, rng, new_size,
                                            config=cfg)


def _task_label(mult, N_, rname, rep):
    return f"mult={mult} N={N_:4d} {rname:18s} rep={rep:02d}"


def _task_label_from_meta(meta):
    return _task_label(meta["mult"], meta["N"], meta["resampler"], meta["rep"])


def _build_dense_lg(d_state, alpha, Q_orig, H_orig, R_orig, eig_spread,
                    seed=2026):
    """
    Build a dense LG model by rotating the state space with a random
    orthogonal matrix P and assigning distinct eigenvalues to A.

    A_dense = P @ diag(lambda_1, ..., lambda_d) @ P^T
    where lambda_j = alpha^{1 + spread * (2j/(d-1) - 1)}  for j=0,...,d-1
    so eigenvalues range from alpha^{1-spread} to alpha^{1+spread}, all < 1.

    Q, H are rotated accordingly.  The Kalman filter receives the dense
    (A_dense, Q_dense, H_dense, R) and works exactly.
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d_state, d_state))
    P, _ = np.linalg.qr(G)

    if d_state == 1:
        eigvals = np.array([alpha])
    else:
        exponents = 1.0 + eig_spread * (2.0 * np.arange(d_state) / (d_state - 1) - 1.0)
        eigvals = alpha ** exponents          # all in (0, 1)

    A_dense = P @ np.diag(eigvals) @ P.T     # dense, symmetric, stable
    Q_dense = P @ (P.T @ Q_orig @ P) @ P.T   # = Q_orig (if Q_orig commutes with P)
    # In general Q_dense should simply remain Q_orig if it's already in the
    # original coordinate system.  But if kappa>1 the model already built Q in
    # the original basis, so we keep it unchanged.
    Q_dense = Q_orig
    H_dense = H_orig                          # observation stays in original coords
    # (Both Q and H operate on the same coordinate system as A_dense; the only
    #  thing that changed is the *dynamics* within that system.)

    return A_dense, Q_dense, H_dense, R_orig, eigvals, P


# Model setup -----------------------------------------------------------------

A_MAT, Q_PROC, H_OBS, R_NOISE = build_lg_model(
    d_state=d, alpha=ALPHA,
    q_scale=q_scale, r_scale=r_scale, kappa=kappa,
)

A_MAT, Q_PROC, H_OBS, R_NOISE, _EIGVALS, _P_BASIS = \
    _build_dense_lg(d, ALPHA, Q_PROC, H_OBS, R_NOISE, eig_spread=0.5)


RESAMPLERS = {
    "Multinomial":  resample_multinomial,
    "Stratified":   resample_stratified,
    "Residual":     resample_residual,
    "Hilbert":      resample_hilbert,
    "WDR":          _resample_wdr,
}

RESAMPLER_COLORS = {
    "Multinomial": "#8D8D8D",
    "Stratified":  "#a6cee3",
    "Residual":    "#1f78b4",
    "Hilbert":     "#fdbf6f",
    "WDR":         "#e31a1c",
}


# Sub-step decomposition -------------------------------------------------------

def build_substep_params(A, Q, k, eigvals, P):
    """
    Decompose  x_{t+1} = A x_t + eps,  eps ~ N(0, Q)
    into k identical sub-steps that are *marginally equivalent*:
        x_{t,i} = A_sub x_{t,i-1} + eps_sub,   eps_sub ~ N(0, Q_sub),  i=1..k

    Uses the pre-computed eigendecomposition A = P diag(eigvals) P^T
    from _build_dense_lg.

    Derivation (eigenbasis):
        A_sub = P D^{1/k} P^T.
        Requirement: sum_{i=0}^{k-1} A_sub^i Q_sub (A_sub^i)^T = Q.
        In eigenbasis with r_{ij} = d_i d_j, the (i,j) entry gives the
        geometric series sum_{l=0}^{k-1} r^{l/k} = (1 - r) / (1 - r^{1/k}),
        hence
            Q_sub_tilde[i,j] = Q_tilde[i,j] * (1 - (d_i d_j)^{1/k}) / (1 - d_i d_j)
        and Q_sub = P Q_sub_tilde P^T.
    """
    if k == 1:
        return A, Q

    # A_sub = P diag(eigvals^{1/k}) P^T
    eigvals_sub = np.sign(eigvals) * np.abs(eigvals) ** (1.0 / k)
    A_sub = P @ np.diag(eigvals_sub) @ P.T

    # Q_sub via element-wise scaling in eigenbasis
    Q_tilde = P.T @ Q @ P
    dd = eigvals[:, None] * eigvals[None, :]                # d_i * d_j
    dd1k = np.sign(dd) * np.abs(dd) ** (1.0 / k)            # (d_i d_j)^{1/k}
    safe = np.abs(1.0 - dd) > 1e-12
    S = np.where(safe, (1.0 - dd1k) / (1.0 - dd), 1.0 / k)

    Q_sub_tilde = Q_tilde * S
    Q_sub = P @ Q_sub_tilde @ P.T
    Q_sub = 0.5 * (Q_sub + Q_sub.T)

    return A_sub, Q_sub


def propagate_lg_substeps(X, A_sub, Q_sub, rng, n_descendants, n_substeps):
    """Propagate particles through n_substeps sub-transitions."""
    N_anc = X.shape[0]
    d_state = A_sub.shape[0]
    X_prop = np.repeat(X, n_descendants, axis=0)
    n_total = N_anc * n_descendants
    for _ in range(n_substeps):
        X_prop = X_prop @ A_sub.T       # dense matmul: O(N * d^2)
        X_prop += rng.multivariate_normal(np.zeros(d_state), Q_sub, size=n_total)
    return X_prop


# Core functions ---------------------------------------------------------------

def loglik_gaussian_obs(y, X, H, R_):
    innov = y - (X @ H.T)
    Rinv = np.linalg.inv(R_)
    quad = np.einsum('ni,ij,nj->n', innov, Rinv, innov)
    return -0.5 * quad


def cv_of(w):
    n = w.size
    return np.sqrt(np.mean((n * w - 1.0) ** 2))


def rmse_over_time(Xhat, Xtrue):
    return np.sqrt(np.mean((Xhat - Xtrue) ** 2, axis=1))


def compute_es(X_samp, W_samp, mu_kf):
    """
    Compute the Energy Score (ES) against the Kalman filter mean per timestep.

    ES_t = E_P[||X - mu_t||] - 0.5 * E_{P,P}[||X - X'||]

    Returns: es, shape (T,)
    """
    T, N, D = X_samp.shape
    if W_samp is None or W_samp.size == 0:
        W = np.full((T, N), 1.0 / N)
    else:
        W = np.asarray(W_samp, float)
        W = W / (W.sum(axis=1, keepdims=True) + 1e-16)

    es = np.empty(T)
    for t in range(T):
        xt, wt = X_samp[t], W[t]
        mu_t = mu_kf[t]

        dist_XX = cdist(xt, xt)                           # (N, N)
        wDw = wt @ dist_XX @ wt                           # scalar

        es_term1 = wt @ cdist(xt, mu_t[None, :]).ravel()  # E_P[||X-mu||]
        es[t] = es_term1 - 0.5 * wDw

    return es


# Timed particle filter -------------------------------------------------------

def particle_filter_timed(Y_obs, N, R_desc, resample_func, seed,
                          n_substeps, pre_hook=None):
    """
    Bootstrap PF for LG with separate timing for propagation+weight vs resampling.

    `pre_hook(t, Xf, w)`, if given, receives the propagated particles and their
    normalized weights at every timestep before resampling.
    """
    rng = np.random.default_rng(seed)
    T = Y_obs.shape[0]
    d_state = A_MAT.shape[0]

    A_sub, Q_sub = build_substep_params(A_MAT, Q_PROC, n_substeps, _EIGVALS, _P_BASIS)
    cv_threshold = CV_THRES if N >= 10 else -1.0

    # Initial particles from prior
    X = rng.multivariate_normal(np.zeros(d_state), Q_PROC, size=N)

    # Propagate once (to match t=0 convention: state after one transition)
    X = propagate_lg_substeps(X, A_sub, Q_sub, rng, n_descendants=1,
                              n_substeps=n_substeps)

    logw0 = loglik_gaussian_obs(Y_obs[0], X, H_OBS, R_NOISE)
    logw0 -= logsumexp(logw0)
    w = np.exp(logw0)

    X_est = np.zeros((T, d_state))  # ensemble mean
    cv = np.zeros(T)
    cv[0] = cv_of(w)
    if pre_hook is not None:
        pre_hook(0, X, w)

    total_prop_time = 0.0
    total_resamp_time = 0.0
    resamp_calls = 0

    # Initial resampling
    if cv[0] > cv_threshold:
        t0 = timer()
        out = resample_func(X, w, rng, N)
        total_resamp_time += timer() - t0
        resamp_calls += 1
        if isinstance(out, tuple):
            X, w = out[0], out[1]
            w /= np.sum(w)
        else:
            X = out.copy()
            w = np.full(N, 1.0 / N)

    X_est[0] = np.average(X, axis=0, weights=w)
    X_samp = [X.copy()]
    w_samp = [w.copy()]
    logw_prev = np.log(w)

    for k in range(1, T):
        R_local = R_desc if cv[k - 1] > cv_threshold else 1

        # --- Propagation + weighting (timed) ---
        t0 = timer()
        Xf = propagate_lg_substeps(X, A_sub, Q_sub, rng,
                                   n_descendants=R_local,
                                   n_substeps=n_substeps)
        loglik = loglik_gaussian_obs(Y_obs[k], Xf, H_OBS, R_NOISE)
        logw = np.repeat(logw_prev, R_local) + loglik
        logw -= logsumexp(logw)
        w = np.exp(logw)
        total_prop_time += timer() - t0

        cv[k] = cv_of(w)
        if pre_hook is not None:
            pre_hook(k, Xf, w)

        # --- Resampling (timed) ---
        if cv[k] > cv_threshold or R_local > 1:
            t0 = timer()
            out = resample_func(Xf, w, rng, N)
            total_resamp_time += timer() - t0
            resamp_calls += 1
            if isinstance(out, tuple):
                X, w_prev = out[0], out[1]
                w_prev /= np.sum(w_prev)
                logw_prev = np.log(w_prev)
            else:
                X = out.copy()
                logw_prev = np.full(N, -np.log(N))
        else:
            X = Xf.copy()
            logw_prev = logw.copy()

        w = np.exp(logw_prev)
        X_est[k] = np.average(X, axis=0, weights=w)
        X_samp.append(X.copy())
        w_samp.append(w.copy())

    X_samp = np.array(X_samp)
    w_samp = np.array(w_samp)

    return X_est, X_samp, w_samp, total_prop_time, total_resamp_time, resamp_calls


# Worker function for parallel execution --------------------------------------

_WORKER_GPU = None
_WORKER_GPU_IDS = None
_WORKER_COUNTER = None
_WORKER_LOCK = None

def _init_worker_gpu(gpu_ids, counter, lock):
    """Store GPU info for lazy initialization — only initialize CUDA when needed."""
    global _WORKER_GPU_IDS, _WORKER_COUNTER, _WORKER_LOCK
    _WORKER_GPU_IDS = gpu_ids
    _WORKER_COUNTER = counter
    _WORKER_LOCK = lock


def _ensure_gpu():
    """Initialize CUDA on first call (lazy), for GPU-using resamplers like WDR."""
    global _WORKER_GPU
    if _WORKER_GPU is not None:
        return
    if _WORKER_LOCK is None or _WORKER_COUNTER is None or _WORKER_GPU_IDS is None:
        gpu = 0
    else:
        with _WORKER_LOCK:
            idx = _WORKER_COUNTER.value
            _WORKER_COUNTER.value += 1
        gpu = _WORKER_GPU_IDS[idx % len(_WORKER_GPU_IDS)]
    torch.cuda.set_device(gpu)
    _ = torch.empty(1, device="cuda")
    _WORKER_GPU = gpu


# Resamplers that require GPU


def _run_one_task(mult, N_, rname, rep, Y_obs, mu_kf, pf_seed):
    """Single (mult, N, resampler, rep) combination."""
    try:
        if rname == "WDR":
            _ensure_gpu()
        rfunc = RESAMPLERS[rname]
        nsteps = mult

        t_pf_start = timer()
        X_est, X_samp, w_samp, t_prop, t_resamp, resamp_calls = particle_filter_timed(
            Y_obs, N_, R, rfunc, pf_seed, nsteps,
        )
        t_pf = timer() - t_pf_start

        rmse_val = float(np.mean(rmse_over_time(X_est, mu_kf)))
        es_t = compute_es(X_samp, w_samp, mu_kf)
        es_val = float(np.mean(es_t))

        return {
            "mult": mult,
            "nsteps": nsteps,
            "N": N_,
            "resampler": rname,
            "rep": rep,
            "prop_time": t_prop,
            "resamp_time": t_resamp,
            "resamp_calls": int(resamp_calls),
            "total_time": t_pf,
            "rmse": rmse_val,
            "es": es_val,
        }
    except BaseException:
        raise


def _std_or_zero(values):
    return np.std(values, ddof=1) if len(values) > 1 else 0.0


def _summarize_group_rows(rows, mult, N_, rname, T_steps):
    prop_times = [r["prop_time"] for r in rows]
    resamp_times = [r["resamp_time"] for r in rows]
    resamp_calls = [r["resamp_calls"] for r in rows]
    total_times = [r["total_time"] for r in rows]
    rmse_vals = [r["rmse"] for r in rows]
    es_vals = [r["es"] for r in rows]

    prop_mean = np.mean(prop_times)
    total_mean = np.mean(total_times)

    return {
        "mult": mult,
        "nsteps": mult,
        "N": N_,
        "resampler": rname,
        "prop_time_mean": prop_mean,
        "prop_time_std": _std_or_zero(prop_times),
        "resamp_time_mean": np.mean(resamp_times),
        "resamp_time_std": _std_or_zero(resamp_times),
        "resamp_calls_mean": np.mean(resamp_calls),
        "resamp_calls_std": _std_or_zero(resamp_calls),
        "total_time_mean": total_mean,
        "total_time_std": _std_or_zero(total_times),
        "prop_frac": prop_mean / (total_mean + 1e-10),
        "prop_per_particle": prop_mean / (N_ * T_steps),
        "prop_per_particle_std": _std_or_zero(prop_times) / (N_ * T_steps),
        "rmse_mean": np.mean(rmse_vals),
        "rmse_std": _std_or_zero(rmse_vals),
        "es_mean": np.mean(es_vals),
        "es_std": _std_or_zero(es_vals),
    }


def main():

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    Ns_tag = "_".join(str(n) for n in Ns)
    out_dir = "outputs_lg_cost"
    run_dir = f"{out_dir}/d{d}_mult{SUBSTEP_MULTIPLIER}_Ns{Ns_tag}_R{R}"
    res_dir = f"{run_dir}/results_mult{SUBSTEP_MULTIPLIER}_{timestamp}"
    os.makedirs(res_dir, exist_ok=True)

    # Save config
    config = dict(
        d=d, alpha=ALPHA, q_scale=q_scale, r_scale=r_scale,
        kappa=kappa, Ns=Ns, R=R, T=T_steps, cv_threshold=CV_THRES,
        n_repeats=n_repeats, substep_multiplier=SUBSTEP_MULTIPLIER,
        dense=True, eig_spread=0.5,
        parallel=if_parallel, max_workers=MAX_WORKERS,
        timestamp=timestamp,
    )
    with open(f"{res_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate fixed dataset + Kalman oracle
    _, Y_obs = simulate_lg_data(T_steps, d, A_MAT, Q_PROC, H_OBS, R_NOISE,
                                seed=12345)
    mu0 = np.zeros(d)
    Sigma0 = Q_PROC.copy()
    mu_kf, _ = kalman_filter(Y_obs, A_MAT, Q_PROC, H_OBS, R_NOISE,
                                         mu0, Sigma0)
    # ====== Run experiments for a single n_substeps multiplier ======
    # Collect per-rep rows, then aggregate into per-(N, resampler) records.
    all_rows = []  # list of per-rep dicts from _run_one_task
    task_specs = []
    for N_ in Ns:
        for rname in RESAMPLERS:
            for rep in range(n_repeats):
                pf_seed = 24680 + rep * 1000 + N_
                task_specs.append(
                    {
                        "mult": SUBSTEP_MULTIPLIER,
                        "N": N_,
                        "resampler": rname,
                        "rep": rep,
                        "pf_seed": pf_seed,
                    }
                )

    if if_parallel:
        real_gpu_ids = [int(x) for x in cuda_no.split(",") if x.strip() != ""]
        gpu_ids = list(range(len(real_gpu_ids))) if real_gpu_ids else [0]
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
        lock = ctx.Lock()
        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            mp_context=ctx,
            initializer=_init_worker_gpu,
            initargs=(gpu_ids, counter, lock),
        ) as ex:
            futures = {}
            try:
                for spec in task_specs:
                    fut = ex.submit(
                        _run_one_task,
                        spec["mult"],
                        spec["N"],
                        spec["resampler"],
                        spec["rep"],
                        Y_obs,
                        mu_kf,
                        spec["pf_seed"],
                    )
                    futures[fut] = spec

                for fut in as_completed(futures):
                    meta = futures[fut]
                    try:
                        row = fut.result()
                    except BrokenProcessPool as exc:
                        raise RuntimeError(
                            f"Process pool broke while running {_task_label_from_meta(meta)}"
                        ) from exc
                    except Exception as exc:
                        raise RuntimeError(
                            f"Task failed: {_task_label_from_meta(meta)}"
                        ) from exc

                    all_rows.append(row)
            except BrokenProcessPool as exc:
                raise RuntimeError("Process pool broke during parallel execution") from exc
    else:
        for N_ in Ns:
            for rname in RESAMPLERS:
                rows_group = []
                for rep in range(n_repeats):
                    pf_seed = 24680 + rep * 1000 + N_
                    row = _run_one_task(
                        SUBSTEP_MULTIPLIER,
                        N_,
                        rname,
                        rep,
                        Y_obs,
                        mu_kf,
                        pf_seed,
                    )
                    all_rows.append(row)
                    rows_group.append(row)

                rec = _summarize_group_rows(
                    rows_group, SUBSTEP_MULTIPLIER, N_, rname, T_steps,
                )

    # Aggregate per-rep rows into per-(N, resampler) records for this multiplier.
    results = []
    for N_ in Ns:
        for rname in RESAMPLERS:
            rows = [r for r in all_rows
                    if r["N"] == N_ and r["resampler"] == rname]
            rec = _summarize_group_rows(
                rows, SUBSTEP_MULTIPLIER, N_, rname, T_steps,
            )
            results.append(rec)

    # Save results (config + records, so plots can be reproduced from pkl alone)
    with open(f"{res_dir}/results.pkl", "wb") as f:
        pickle.dump({"config": config, "results": results},
                    f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
