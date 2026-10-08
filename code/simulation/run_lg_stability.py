"""Between-run stability analysis (Figure 3).

Fixes one dataset, runs the particle filter multiple times with different
seeds, and measures between-run stability as a function of N with the three
metrics of Figure 3: the particle energy distance between runs, the estimate
standard deviation, and the RMSE standard deviation.

Usage: see code/experiments/fig3_lg_stability.sh
"""
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Between-run stability analysis.")
    parser.add_argument("--d", dest="d", default=50, type=int, help="State dimension")
    parser.add_argument("--N", dest="N_list", default="256,512,1024,2048", type=str,
                        help="Comma-separated list of particle counts")
    parser.add_argument("--R", dest="R", default=3, type=int, help="Number of descendants")
    parser.add_argument("--n-runs", dest="n_runs", default=20, type=int, help="Number of independent PF runs")
    parser.add_argument("--T", dest="T", default=100, type=int, help="Number of timesteps")
    parser.add_argument("--cuda", dest="cuda_no", default="0,1,2,3", type=str,
                        help="CUDA device ids for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--parallel", dest="if_parallel", default=1, type=int,
                        help="Whether to run the filter runs in parallel")
    parser.add_argument("--cv", dest="cv_threshold", default=2, type=float,
                        help="Weight-CV threshold that triggers resampling")
    parser.add_argument("--workers", dest="max_workers", default=10, type=int,
                        help="Max parallel worker processes")
    return parser.parse_args()

args = parse_args()
d_lg = args.d
N_list = [int(x.strip()) for x in args.N_list.split(",")]
R = args.R
n_runs = args.n_runs
T_steps = args.T
cuda_no = args.cuda_no
if_parallel = args.if_parallel
CV_THRES = args.cv_threshold
MAX_WORKERS = args.max_workers

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_no
# Limit BLAS threads per process to avoid oversubscription with many workers
os.environ['OMP_NUM_THREADS'] = '64'
os.environ['OPENBLAS_NUM_THREADS'] = '64'
os.environ['MKL_NUM_THREADS'] = '64'

import numpy as np
import time
from scipy.special import logsumexp
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


# Resampler definitions -------------------------------------------------------
RESAMPLERS = {
    "Multinomial":  resample_multinomial,
    "Stratified":   resample_stratified,
    "Residual":     resample_residual,
    "Hilbert":      resample_hilbert,
    "WDR":          wasserstein_dirichlet_resampling,
}

RESAMPLER_SEED_OFFSETS = {
    rname: 137 * idx for idx, rname in enumerate(RESAMPLERS.keys())
}


# Model-specific setup -------------------------------------------------------

def cv_of(w):
    n = w.size
    return np.sqrt(np.mean((n * w - 1.0) ** 2))

def loglik_gaussian_obs(y, X, H, R_):
    innov = y - (X @ H.T)
    Rinv = np.linalg.inv(R_)
    quad = np.einsum('ni,ij,nj->n', innov, Rinv, innov)
    return -0.5 * quad


def weighted_mean(X, w):
    w_norm = w / (np.sum(w) + 1e-16)
    return np.einsum('ni,n->i', X, w_norm)


def rmse_over_time(Xhat, Xref):
    return np.sqrt(np.mean((Xhat - Xref) ** 2, axis=1))


def std_with_ddof1(values, axis=0):
    values = np.asarray(values)
    if values.shape[axis] <= 1:
        out_shape = values.shape[:axis] + values.shape[axis + 1:]
        return np.zeros(out_shape, dtype=float)
    return np.std(values, axis=axis, ddof=1)


def estimate_dispersion_over_time(est_runs):
    """
    est_runs: (n_runs, T, d)
    Returns per-time dispersion:
        sqrt( mean_j Var_r[ xhat_t^{(r)}[j] ] )
    """
    var_tj = std_with_ddof1(est_runs, axis=0) ** 2
    return np.sqrt(np.mean(var_tj, axis=1))


def rmse_std_over_time(est_runs, ref_path):
    """
    est_runs: (n_runs, T, d)
    ref_path: (T, d)
    Returns per-time std across runs of RMSE-to-reference.
    """
    rmse_runs = np.sqrt(np.mean((est_runs - ref_path[None, :, :]) ** 2, axis=2))
    return std_with_ddof1(rmse_runs, axis=0)


# ---- LG Model ----
d_state = d_lg
ALPHA = 0.95
A_MAT, Q_PROC, H_OBS, R_NOISE = build_lg_model(
    d_state=d_state,
    alpha=ALPHA,
    kappa=50.0,
    q_scale=0.5,
    r_scale=1.0,
)

def propagate(X, R_local, rng):
    return propagate_lg_particles(X, A_MAT, Q_PROC, rng, n_descendants=R_local)

def simulate_data_lg(T, seed=42):
    return simulate_lg_data(T, d_state, A_MAT, Q_PROC, H_OBS, R_NOISE, seed)

def init_particles(N, rng):
    return rng.multivariate_normal(np.zeros(d_state), Q_PROC, size=N)


# Generic particle filter --------------------------------------------------

def particle_filter_generic(
    Y_obs, N, R_desc, resample_func, seed, cv_threshold=CV_THRES,
    ed_timesteps=None,
):
    """Run the filter, returning the estimate and, at the
    timesteps listed in `ed_timesteps`, the particle cloud
    with its weights (used for the between-run energy distance)."""
    rng = np.random.default_rng(seed)
    T = Y_obs.shape[0]
    ed_wanted = set() if ed_timesteps is None else {int(t) for t in ed_timesteps}
    clouds = {}

    cv_threshold = cv_threshold if N >= 10 else -1.0
    X0 = init_particles(N, rng)
    Xf0 = propagate(X0, 1, rng)

    logw0 = loglik_gaussian_obs(Y_obs[0], Xf0, H_OBS, R_NOISE)
    logw0 -= logsumexp(logw0)
    w = np.exp(logw0)

    cv = np.zeros(T)
    cv[0] = cv_of(w)
    est = np.zeros((T, Xf0.shape[1]))  # ensemble mean

    if cv[0] > cv_threshold:
        out = resample_func(Xf0, w, rng, N)
        if isinstance(out, tuple):
            X = out[0]
            w = out[1]
            w /= np.sum(w)
        else:
            X = out.copy()
            w = np.full(N, 1.0 / N)
    else:
        X = Xf0.copy()

    est[0] = weighted_mean(X, w)
    if 0 in ed_wanted:
        clouds[0] = (X.copy(), w.copy())
    logw_prev = np.log(w)

    for k in range(1, T):
        R_local = R_desc if cv[k-1] > cv_threshold else 1

        Xf = propagate(X, R_local, rng)

        loglik = loglik_gaussian_obs(Y_obs[k], Xf, H_OBS, R_NOISE)
        logw = np.repeat(logw_prev, R_local) + loglik
        logw -= logsumexp(logw)
        w = np.exp(logw)

        cv[k] = cv_of(w)

        if cv[k] > cv_threshold or R_local > 1:
            out = resample_func(Xf, w, rng, N)
            if isinstance(out, tuple):
                X = out[0]
                w_prev = out[1]
                w_prev /= np.sum(w_prev)
                logw_prev = np.log(w_prev)
            else:
                X = out.copy()
                logw_prev = np.full(N, -np.log(N))
        else:
            X = Xf.copy()
            logw_prev = logw.copy()

        w = np.exp(logw_prev)
        est[k] = weighted_mean(X, w)
        if k in ed_wanted:
            clouds[k] = (X.copy(), w.copy())

    return est, [clouds[t] for t in sorted(clouds)]


# Single-run worker -----------------------------------------------------------

def _run_one(run_id, rname, Y_obs, N, pf_seed, ed_timesteps):
    if rname == "WDR":
        _ensure_gpu()
    rfunc = RESAMPLERS[rname]
    est, clouds = particle_filter_generic(
        Y_obs, N, R, rfunc, pf_seed, ed_timesteps=ed_timesteps
    )

    return {
        "run_id": run_id,
        "resampler": rname,
        "est": est,
        "clouds": clouds,
    }


# Between-run energy distance --------------------------------------------------

def _batch_weighted_dist(X, w, Y, wy):
    """
    Batch-compute  sum_t  w[t] @ D[t] @ wy[t]  where D[t]_ij = ||X[t,i] - Y[t,j]||.

    X:  (T, N1, d)
    w:  (T, N1)
    Y:  (T, N2, d)
    wy: (T, N2)
    Returns: scalar (sum over T).

    Uses  ||x-y|| = sqrt(||x||^2 + ||y||^2 - 2 x·y)  to avoid per-timestep cdist.
    """
    # (T, N1)  and  (T, N2)
    sq_X = np.einsum('tni,tni->tn', X, X)
    sq_Y = np.einsum('tni,tni->tn', Y, Y)
    # (T, N1, N2) distance matrix squared
    gram = np.matmul(X, Y.transpose(0, 2, 1))          # (T, N1, N2)
    D_sq = sq_X[:, :, None] + sq_Y[:, None, :] - 2.0 * gram
    np.maximum(D_sq, 0.0, out=D_sq)
    D = np.sqrt(D_sq)
    # weighted sum: w[t] @ D[t] @ wy[t]  for each t, then sum over t
    # => einsum('tn,tnm,tm->t', w, D, wy).sum()
    Dwy = np.einsum('tnm,tm->tn', D, wy)               # (T, N1)
    return np.einsum('tn,tn->', w, Dwy)


def between_run_energy_distance(cloud_list, max_pairs=500, seed=42):
    """
    True energy distance between weighted particle distributions from different runs.

    ED(P, Q) = 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||

    cloud_list: list of runs, each run is a list of (particles, weights) tuples.
    Returns: mean energy distance across (subsampled run pairs, timesteps)
    """
    rng = np.random.default_rng(seed)
    n_runs_local = len(cloud_list)
    n_t = len(cloud_list[0])

    all_pairs = [(i, j) for i in range(n_runs_local) for j in range(i+1, n_runs_local)]
    if len(all_pairs) > max_pairs:
        sel = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[k] for k in sel]
    else:
        pairs = all_pairs

    # Pack cloud_list into (n_runs, T, N, d) and (n_runs, T, N) arrays
    N_particles = cloud_list[0][0][0].shape[0]
    d_dim = cloud_list[0][0][0].shape[1]
    X_all = np.empty((n_runs_local, n_t, N_particles, d_dim))
    W_all = np.empty((n_runs_local, n_t, N_particles))
    for r in range(n_runs_local):
        for tidx in range(n_t):
            X_all[r, tidx] = cloud_list[r][tidx][0]
            w = cloud_list[r][tidx][1]
            W_all[r, tidx] = w / (w.sum() + 1e-16)

    # Precompute self-terms per run: sum_t w[t] @ D_self[t] @ w[t]
    self_sums = np.empty(n_runs_local)
    for r in range(n_runs_local):
        self_sums[r] = _batch_weighted_dist(X_all[r], W_all[r], X_all[r], W_all[r])

    # Cross-terms per pair
    ed_vals = np.empty(len(pairs))
    for k, (i, j) in enumerate(pairs):
        cross = _batch_weighted_dist(X_all[i], W_all[i], X_all[j], W_all[j])
        ed_vals[k] = (2.0 * cross - self_sums[i] - self_sums[j]) / n_t

    return float(np.mean(ed_vals))


# Main -----------------------------------------------------------------------

def main():

    # Setup output dirs
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_label = f"LG_d{d_lg}_R{R}"
    out_dir = "outputs_between_run"
    res_dir = f"{out_dir}/{out_label}/results_{timestamp}"
    os.makedirs(res_dir, exist_ok=True)

    # 1) Generate one fixed dataset
    data_seed = 12345
    _, Y_obs = simulate_data_lg(T=T_steps, seed=data_seed)
    mu0 = np.zeros(d_state)
    Sigma0 = Q_PROC.copy()
    ref_path, _ = kalman_filter(Y_obs, A_MAT, Q_PROC, H_OBS, R_NOISE, mu0, Sigma0)

    ed_n_sample_t = min(50, T_steps)
    ed_rng = np.random.default_rng(777)
    ed_timesteps = tuple(
        int(t) for t in np.sort(ed_rng.choice(T_steps, size=ed_n_sample_t, replace=False)))

    base_pf_seed = 99999
    rnames = list(RESAMPLERS.keys())

    # 2) Loop over N
    # results_all[N][rname] = {"br_ed": ..., "est_dispersion_mean": ..., ...}
    results_all = {}
    per_run_estimates = {}  # {N: {rname: {"est_runs": ...}}}

    for N_particles in N_list:
        all_est = {rn: {} for rn in rnames}
        all_clouds = {rn: {} for rn in rnames}

        if if_parallel:
            ctx = mp.get_context("spawn")
            real_gpu_ids = [int(x) for x in cuda_no.split(",") if x.strip() != ""]
            gpu_ids = list(range(max(1, len(real_gpu_ids))))
            counter = ctx.Value('i', 0)
            lock = ctx.Lock()

            with ProcessPoolExecutor(
                max_workers=MAX_WORKERS, mp_context=ctx,
                initializer=_init_worker_gpu, initargs=(gpu_ids, counter, lock),
            ) as ex:
                futures = {}
                for run_id in range(n_runs):
                    for rname in rnames:
                        pf_seed = base_pf_seed + run_id * 1000 + RESAMPLER_SEED_OFFSETS[rname]
                        fut = ex.submit(_run_one, run_id, rname, Y_obs, N_particles,
                                        pf_seed, ed_timesteps)
                        futures[fut] = (run_id, rname)

                for fut in as_completed(futures):
                    rec = fut.result()
                    rn = rec["resampler"]
                    all_est[rn][rec["run_id"]] = rec["est"]
                    all_clouds[rn][rec["run_id"]] = rec["clouds"]
        else:
            for run_id in range(n_runs):
                for rname in rnames:
                    pf_seed = base_pf_seed + run_id * 1000 + RESAMPLER_SEED_OFFSETS[rname]
                    rec = _run_one(run_id, rname, Y_obs, N_particles, pf_seed,
                                   ed_timesteps)
                    rn = rec["resampler"]
                    all_est[rn][rec["run_id"]] = rec["est"]
                    all_clouds[rn][rec["run_id"]] = rec["clouds"]

        stats_N = {}
        est_runs_N = {}
        for rn in rnames:
            est_runs = np.stack([all_est[rn][r] for r in sorted(all_est[rn])], axis=0)
            br_ed = between_run_energy_distance(
                [all_clouds[rn][r] for r in sorted(all_clouds[rn])])
            del all_clouds[rn]
            est_dispersion_t = estimate_dispersion_over_time(est_runs)
            rmse_std_t = rmse_std_over_time(est_runs, ref_path)
            stats_N[rn] = {
                "est_dispersion_t": est_dispersion_t,
                "rmse_std_t": rmse_std_t,
                "est_dispersion_mean": float(np.mean(est_dispersion_t)),
                "rmse_std_mean": float(np.mean(rmse_std_t)),
                "br_ed": br_ed,
            }
            est_runs_N[rn] = {
                "est_runs": est_runs,
            }
        results_all[N_particles] = stats_N
        per_run_estimates[N_particles] = est_runs_N

        # Print summary for this N
        print(f"\n--- N = {N_particles} ---")
        print(
            f"{'Method':<18s} {'BR-ED':>12s} {'est-std':>13s} {'RMSE std':>14s}"
        )
        print("-" * 62)
        for rn in rnames:
            s = stats_N[rn]
            print(
                f"{rn:<18s} "
                f"{s['br_ed']:12.4e} "
                f"{s['est_dispersion_mean']:13.4e} "
                f"{s['rmse_std_mean']:14.4e} "
            )

    # 3) Save results
    payload = {
        "model": "LG",
        "N_list": N_list,
        "R": R,
        "n_runs": n_runs,
        "T": T_steps,
        "ref_path": ref_path,
        "results_all": results_all,
        "per_run_estimates": per_run_estimates,
        "ed_timesteps": ed_timesteps,
    }
    with open(f"{res_dir}/results.pkl", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
