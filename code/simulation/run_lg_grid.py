"""Linear Gaussian model over dimensions and condition numbers (Figure 2).

Compares the classical resamplers, Hilbert resampling, and WDR against the
Kalman filter oracle over a (d, kappa, m) grid; one process per replication.
"""
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Linear Gaussian model with Kalman filter oracle.")
    parser.add_argument("--d", dest="d", default=10, type=int, help="State dimension")
    parser.add_argument("--R", dest="R", default=3, type=int, help="Number of descendants")
    parser.add_argument("--rep", dest="n_repeats", default=20, type=int, help="Number of replications")
    parser.add_argument("--cuda", dest="cuda_no", default="0,1,2,3", type=str, help="CUDA device number")
    parser.add_argument("--parallel", dest="if_parallel", default=1, type=int, help="Whether to run in parallel")
    parser.add_argument("--cv", dest="cv_threshold", default=2, type=float, help="CV threshold")
    parser.add_argument("--workers", dest="max_workers", default=20, type=int, help="Max workers")
    parser.add_argument("--q-scale", dest="q_scale", default=0.5, type=float, help="Process noise scale (default 0.5)")
    parser.add_argument("--r-scale", dest="r_scale", default=1.0, type=float, help="Observation noise scale (default 1.0)")
    parser.add_argument("--kappa", dest="kappa", default=1.0, type=float,
                        help="Condition number of process noise covariance (default 1.0 = isotropic)")
    parser.add_argument("--methods", dest="methods", default="all", type=str,
                        help="Comma-separated resampler names to run (default: all)")
    parser.add_argument("--Ts", dest="ts_list", default="500,1000", type=str,
                        help="Comma-separated time horizons; T = last entry (default 500,1000)")
    parser.add_argument("--Ns", dest="ns_list", default="256,512,1024,2048,4096", type=str,
                        help="Comma-separated particle sizes m (default 256,...,4096)")
    return parser.parse_args()

args = parse_args()
d = args.d
R = args.R
n_repeats = args.n_repeats
cuda_no = args.cuda_no
if_parallel = args.if_parallel
CV_THRES = args.cv_threshold
MAX_WORKERS = args.max_workers
q_scale = args.q_scale
r_scale = args.r_scale
kappa = args.kappa
METHODS = args.methods
TS_LIST = args.ts_list
NS_LIST = args.ns_list

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_no

import numpy as np
from typing import Callable, Tuple
import time
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import multiprocessing as mp
import torch

from lg_common import (
    build_lg_model,
    kalman_filter,
    propagate_lg_particles,
    simulate_lg_data,
)
from ot_resampling import (
    resample_multinomial, resample_stratified, resample_residual,
    wasserstein_dirichlet_resampling,
)
from hilbert_resampling import resample_hilbert

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
    if _WORKER_LOCK is None:  # serial mode: no worker pool to round-robin over
        torch.cuda.set_device(0)
        _ = torch.empty(1, device="cuda")
        _WORKER_GPU = 0
        return
    with _WORKER_LOCK:
        idx = _WORKER_COUNTER.value
        _WORKER_COUNTER.value += 1
    gpu = _WORKER_GPU_IDS[idx % len(_WORKER_GPU_IDS)]
    torch.cuda.set_device(gpu)
    _ = torch.empty(1, device="cuda")
    _WORKER_GPU = gpu


# Model setup -----------------------------------------------------------------

# Linear Gaussian state-space model:
#   X^(t) = A X^(t-1) + xi^(t),   xi ~ N(0, Q)
#   Y^(t) = H X^(t)   + eps^(t),  eps ~ N(0, R_obs)


ALPHA = 0.95  # AR(1) coefficient
A_MAT, Q_PROC, H_OBS, R_NOISE = build_lg_model(
    d_state=d,
    alpha=ALPHA,
    q_scale=q_scale,
    r_scale=r_scale,
    kappa=kappa,
)

# Experiment config
Ts = [int(x) for x in TS_LIST.split(",") if x.strip()]
Ns = [int(x) for x in NS_LIST.split(",") if x.strip()]
base_seed_data = 12345
base_seed_pf   = 24680

# Output directories
import json


def build_output_paths() -> Tuple[str, str, str]:
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    out_dir = "outputs_lg"
    exp_dir = f"{out_dir}/d{d}_kappa{kappa}_r{r_scale}_R{R}_cv{CV_THRES:g}"
    res_dir = f"{exp_dir}/results_{timestamp}"
    return exp_dir, res_dir


def prepare_output_dirs() -> str:
    exp_dir, res_dir = build_output_paths()
    os.makedirs(res_dir, exist_ok=True)

    config = dict(
        d=d, alpha=ALPHA, q_scale=q_scale, r_scale=r_scale,
        kappa=kappa, R=R, cv_threshold=CV_THRES, Ts=Ts, Ns=Ns,
        n_repeats=n_repeats, parallel=if_parallel,
        methods=METHODS,
    )
    with open(f"{exp_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    return res_dir


# Data simulation -------------------------------------------------------------

def simulate_data(T: int, d_state: int, seed: int = 42
                  ) -> Tuple[np.ndarray, np.ndarray]:
    return simulate_lg_data(T, d_state, A_MAT, Q_PROC, H_OBS, R_NOISE, seed)


# Likelihood ------------------------------------------------------------------

def loglik_gaussian_obs(y: np.ndarray, X: np.ndarray, H: np.ndarray, R_: np.ndarray) -> np.ndarray:
    innov = y - (X @ H.T)
    Rinv = np.linalg.inv(R_)
    quad = np.einsum('ni,ij,nj->n', innov, Rinv, innov)
    return -0.5 * quad


def cv_of(w: np.ndarray) -> float:
    n = w.size
    cv2 = np.mean((n * w - 1.0) ** 2)
    return np.sqrt(cv2)


# Particle filter -------------------------------------------------------------

def particle_filter(
    Y_obs: np.ndarray,
    N: int = 20,
    R: int = 3,
    resample: Callable = resample_stratified,
    seed: int = 2025,
    cv_threshold: float = CV_THRES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap particle filter for the linear Gaussian model.
    Propagation: X_new = A X_old + noise (N(0, Q))
    Observation: log p(y|x) = -0.5 (y-Hx)^T R^{-1} (y-Hx)
    """
    rng = np.random.default_rng(seed)
    T = Y_obs.shape[0]
    d_state = A_MAT.shape[0]

    cv_threshold = cv_threshold if N >= 10 else -1.0

    # X_true[0] is generated after one transition from the prior, so we must
    # propagate once before weighting with Y_obs[0].
    X_prev = rng.multivariate_normal(mean=np.zeros(d_state), cov=Q_PROC, size=N)
    Xf0 = propagate_lg_particles(X_prev, A_MAT, Q_PROC, rng, n_descendants=1)

    # Initial weighting at t=0
    logw0 = loglik_gaussian_obs(Y_obs[0], Xf0, H_OBS, R_NOISE)
    logw0 -= logsumexp(logw0)
    w = np.exp(logw0)

    X_est = np.zeros((T, d_state))  # ensemble mean
    cv = np.zeros(T)
    cv[0] = cv_of(w)

    if cv[0] > cv_threshold:
        out = resample(Xf0, w, rng, N)
        if isinstance(out, tuple):
            X, w = out[0], out[1]
            w /= np.sum(w)
        else:
            X = out.copy()
            w = np.full(N, 1.0 / N)
    else:
        X = Xf0.copy()

    X_est[0] = np.average(X, axis=0, weights=w)

    X_samp = [X.copy()]
    w_samp = [w.copy()]

    logw_prev = np.log(w)

    for k in range(1, T):
        if cv[k-1] > cv_threshold:
            R_local = R
        else:
            R_local = 1

        # 1) Propagate: X_new = A * X_old + noise
        Xf = propagate_lg_particles(X, A_MAT, Q_PROC, rng, n_descendants=R_local)

        # 2) Weight
        loglik = loglik_gaussian_obs(Y_obs[k], Xf, H_OBS, R_NOISE)
        logw = np.repeat(logw_prev, R_local) + loglik
        logw -= logsumexp(logw)
        w = np.exp(logw)

        # 3) CV
        cv[k] = cv_of(w)

        # 4) Resample
        if cv[k] > cv_threshold or R_local > 1:
            out = resample(Xf, w, rng, N)
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

        # 5) Estimate: ensemble mean
        X_est[k] = np.average(X, axis=0, weights=w)

        X_samp.append(X.copy())
        w_samp.append(w.copy())

    return X_est, np.array(X_samp), np.array(w_samp)


# Metrics ---------------------------------------------------------------------

def rmse_over_time(Xhat: np.ndarray, Xtrue: np.ndarray) -> np.ndarray:
    diff2 = (Xhat - Xtrue) ** 2
    return np.sqrt(np.mean(diff2, axis=1))

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


# Resampler definitions -------------------------------------------------------

RESAMPLERS = {
    "Multinomial":  resample_multinomial,
    "Stratified":   resample_stratified,
    "Residual":     resample_residual,
    "Hilbert":      resample_hilbert,
    "WDR":          wasserstein_dirichlet_resampling,
}

if METHODS != "all":
    _selected = [s.strip() for s in METHODS.split(",") if s.strip()]
    _unknown = [s for s in _selected if s not in RESAMPLERS]
    if _unknown:
        raise ValueError(f"Unknown methods {_unknown}; available: {list(RESAMPLERS)}")
    RESAMPLERS = {k: RESAMPLERS[k] for k in _selected}


# Worker function -------------------------------------------------------------

def _run_one(rep: int, N_: int, R: int, rname: str,
             Y_obs: np.ndarray,
             mu_kf: np.ndarray,
             pf_seed: int) -> dict:
    """Single (rep, N, resampler) combination."""
    if rname == "WDR":
        _ensure_gpu()
    rfunc = RESAMPLERS[rname]

    X_est, X_samp, w_samp = particle_filter(
        Y_obs=Y_obs, N=N_, R=R,
        resample=rfunc, seed=pf_seed,
        cv_threshold=CV_THRES,
    )

    rmse = rmse_over_time(X_est, mu_kf)
    es = compute_es(X_samp, w_samp, mu_kf)

    return {
        "rep": rep,
        "N": int(N_),
        "R": int(R),
        "resampler": rname,
        "seed": int(pf_seed),
        "rmse": rmse, "es": es,
    }


# Main ------------------------------------------------------------------------

def main():
    res_dir = prepare_output_dirs()

    # Generate data and run Kalman filter on the fly (both are fast)
    scenarios = []
    for rep in range(1, n_repeats + 1):
        data_seed = base_seed_data + (rep - 1)
        _, Y_obs = simulate_data(T=Ts[-1], d_state=d, seed=data_seed)

        # Kalman filter (exact oracle)
        mu0 = np.zeros(d)
        Sigma0 = Q_PROC.copy()
        mu_kf, _ = kalman_filter(Y_obs, A_MAT, Q_PROC, H_OBS, R_NOISE, mu0, Sigma0)

        scenarios.append((rep, Y_obs, mu_kf))


    # Result containers
    rows_runs = []
    metrics = ["rmse", "es"]
    metrics_N = {m: {rn: {N_: [] for N_ in Ns} for rn in RESAMPLERS} for m in metrics}


    if if_parallel:
        ctx = mp.get_context("spawn")
        real_gpu_ids = [int(x) for x in cuda_no.split(",") if x.strip() != ""]
        if len(real_gpu_ids) == 0:
            gpu_ids = [0]
        else:
            gpu_ids = list(range(len(real_gpu_ids)))

        counter = ctx.Value('i', 0)
        lock = ctx.Lock()

        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            mp_context=ctx,
            initializer=_init_worker_gpu,
            initargs=(gpu_ids, counter, lock),
        ) as ex:
            tasks = []
            for (rep, Y_obs, mu_kf) in scenarios:
                for N_ in Ns:
                    for rname in RESAMPLERS.keys():
                        pf_seed = base_seed_pf + rep * 10000 + N_
                        tasks.append(
                            ex.submit(
                                _run_one, rep, N_, R, rname,
                                Y_obs, mu_kf,
                                pf_seed,
                            )
                        )

            for fut in as_completed(tasks):
                rec = fut.result()
                rows_runs.append(rec)
                for m in metrics:
                    metrics_N[m][rec["resampler"]][rec["N"]].append(rec[m])
    else:
        for (rep, Y_obs, mu_kf) in scenarios:
            for N_ in Ns:
                for rname in RESAMPLERS.keys():
                    pf_seed = base_seed_pf + rep * 10000 + N_
                    rec = _run_one(
                        rep, N_, R, rname,
                        Y_obs, mu_kf,
                        pf_seed,
                    )
                    rows_runs.append(rec)
                    for m in metrics:
                            metrics_N[m][rec["resampler"]][rec["N"]].append(rec[m])

    # Save results
    payload = dict(
        d=d, kappa=kappa, Ns=Ns, T=Ts[-1],
        metrics_N=metrics_N, rows_runs=rows_runs,
    )
    with open(f"{res_dir}/results.pkl", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"d={d} kappa={kappa:g}: {len(rows_runs)} runs -> {res_dir}/results.pkl",
          flush=True)


if __name__ == "__main__":
    main()
