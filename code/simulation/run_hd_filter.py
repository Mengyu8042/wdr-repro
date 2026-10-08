"""Driver for the high-dimensional spatial model of Figure 5."""
import argparse
import json
import os
import numpy as np

from hd_model import (build_model, simulate, kf_mean_timevarying,
                      steady_state_covariance, block_filter, metrics,
                      EnergyAccumulator, make_blocks)
from ot_resampling import (resample_multinomial, resample_stratified,
                                 resample_residual)
from hilbert_resampling import resample_hilbert
DATA_SEED_BASE = 74001
PF_SEED_BASE = 84001


def make_wdr_batched(seed):
    """Batched blockwise WDR (see hd_wdr_batched)."""
    import torch
    import hd_wdr_batched as bw
    dev = torch.device("cuda")
    torch.manual_seed(seed + 777_777)
    torch.cuda.manual_seed_all(seed + 777_777)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)

    def _r(Xb, W, n_out):
        B, n_src, _ = Xb.shape
        # chunk the block batch so the (chunk, n_src, n_tgt) cost tensor fits:
        # at m=4096 a single 81-block batch would need ~16 GB.
        budget = 3.0e8                       # elements per chunk
        per_blk = max(1.0, n_src * float(n_out) * 0.3)  # sparsified target fraction
        step = max(1, min(B, int(budget / per_blk)))
        outs = []
        for s0 in range(0, B, step):
            xt = torch.tensor(Xb[s0:s0 + step], dtype=torch.float32, device=dev)
            wt = torch.tensor(W[s0:s0 + step], dtype=torch.float32, device=dev)
            o = bw.wdr_blocks(xt, wt, int(n_out), gen, bw.CANON_CFG)
            outs.append(o.double().cpu().numpy())
            del xt, wt, o
        out = np.concatenate(outs, axis=0)
        if not np.isfinite(out).all():
            raise RuntimeError("batched WDR non-finite")
        return out
    return _r


# WDR is not registered here: blockwindow routes it to make_wdr_batched.
RESAMPLERS = {"Multinomial": lambda s: resample_multinomial,
              "Stratified": lambda s: resample_stratified,
              "Residual": lambda s: resample_residual,
              "Hilbert": lambda s: resample_hilbert}


def get_scenario(model, args, rep):
    """Truth trajectory, observations and exact-KF oracle for one replicate.
    Cached on disk so the O(d^3) Kalman recursion is computed once per
    replicate instead of once per (method, m) case."""
    cache = getattr(args, "oracle_cache", None)
    path = None
    if cache:
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, f"scn_n{args.n}_of{args.obs_frac}"
                                   f"_T{args.T}_rep{rep}.npz")
        if os.path.exists(path):
            z = np.load(path)
            return z["X"], z["Y"], z["mu"]
    X, Y = simulate(model, T=args.T, seed=DATA_SEED_BASE + rep)
    mu = kf_mean_timevarying(model, Y, device=args.device)
    if path:
        tmp = f"{path}.{os.getpid()}.tmp.npz"   # numpy appends .npz otherwise
        np.savez(tmp, X=X, Y=Y, mu=mu)
        os.replace(tmp, path)
    return X, Y, mu


def get_sigma(model, args):
    """Steady-state filtering covariance for the energy distance, computed
    once per process (and cached on disk next to the scenarios when
    --oracle-cache is given)."""
    cache = getattr(args, "oracle_cache", None)
    path = None
    if cache:
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, f"sigma_n{args.n}_of{args.obs_frac}.npz")
        if os.path.exists(path):
            return np.load(path)["Sigma"]
    Sigma, iters, delta = steady_state_covariance(model, device=args.device)
    print(f"steady-state covariance: {iters} Riccati iterations, "
          f"final relative change {delta:.1e}", flush=True)
    if path:
        tmp = f"{path}.{os.getpid()}.tmp.npz"
        np.savez(tmp, Sigma=Sigma)
        os.replace(tmp, path)
    return Sigma


def mode_blockwindow(args):
    model = build_model(n=args.n, nu=args.nu, rho=args.rho, alpha=args.alpha,
                        q_scale=args.q, r_scale=args.r, obs_frac=args.obs_frac)
    os.makedirs(args.out, exist_ok=True)
    Sigma = get_sigma(model, args) if args.ed else None
    blocks = make_blocks(args.n, args.block) if args.ed else None
    for m in args.ms:
        for rep in args.reps:
            X = Y = mu = None
            for method in args.methods:
                path = os.path.join(
                    args.out, f"{method}_b{args.block}_m{m}_rep{rep}.json")
                if os.path.exists(path):
                    continue
                if X is None:
                    X, Y, mu = get_scenario(model, args, rep)
                seed = PF_SEED_BASE + rep * 10000 + m
                acc = None
                if args.es or args.ed:
                    acc = EnergyAccumulator(
                        {"oracle": mu}, t_start=args.T // 2,
                        gauss_mu=mu if args.ed else None, gauss_Sigma=Sigma,
                        blocks=blocks)
                if method == "WDR":
                    res = block_filter(model, Y, m=m, seed=seed,
                                       block=args.block, radius=args.radius,
                                       batched_resampler=make_wdr_batched(seed),
                                       es_accum=acc)
                else:
                    res = block_filter(model, Y, m=m, seed=seed,
                                       block=args.block, radius=args.radius,
                                       resampler=RESAMPLERS[method](seed),
                                       es_accum=acc)
                rec = dict(method=method, m=m, rep=rep, T=args.T, d=model.d,
                           block=args.block, block_dim=res["block_dim"],
                           obs_frac=args.obs_frac,
                           **metrics(res["est"], mu, args.T))
                if acc is not None:
                    summ = acc.summary()
                    if args.es:
                        rec["es_oracle"] = summ["es_oracle"]
                    if args.ed:
                        rec["ed"] = summ["ed"]
                json.dump(rec, open(path, "w"), indent=1)
                es_txt = (f" es={rec['es_oracle']:.5f}" if args.es else "")
                ed_txt = (f" ed={rec['ed']:.5f}" if args.ed else "")
                print(f"{method} b={args.block} m={m} rep={rep}: "
                      f"rmse={rec['rmse_oracle']:.5f}{es_txt}{ed_txt}",
                      flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Block particle filter on the spatial model (Figure 5).")
    ap.add_argument("--n", type=int, default=45,
                    help="lattice side; the state dimension is d = n^2")
    ap.add_argument("--nu", type=float, default=1.5,
                    help="Matern smoothness of the innovation covariance")
    ap.add_argument("--rho", type=float, default=6.0,
                    help="Matern range of the innovation covariance")
    ap.add_argument("--alpha", type=float, default=0.95,
                    help="AR(1) coefficient of the dynamics")
    ap.add_argument("--q", type=float, default=0.5,
                    help="process noise scale (Q = q * Matern)")
    ap.add_argument("--r", type=float, default=1.0,
                    help="observation noise variance")
    ap.add_argument("--obs-frac", type=float, default=1.0,
                    help="fraction of lattice sites observed")
    ap.add_argument("--ms", type=int, nargs="+", default=[256, 512, 1024],
                    help="particle sizes m to run")
    ap.add_argument("--T", type=int, default=300,
                    help="number of timesteps")
    ap.add_argument("--n-runs", dest="n_runs", type=int, default=5,
                    help="number of replications, run as 1..n")
    ap.add_argument("--methods", nargs="+",
                    default=["Multinomial", "Stratified", "Residual", "Hilbert", "WDR"],
                    help="resamplers to run")
    ap.add_argument("--device", default="cuda",
                    help="device for the Kalman oracle recursions")
    ap.add_argument("--block", type=int, default=5,
                    help="block side length of the block particle filter")
    ap.add_argument("--radius", type=float, default=None,
                    help="localization radius of the block observation sets "
                         "(default: the block size)")
    ap.add_argument("--oracle-cache", default=None,
                    help="directory caching the per-replicate scenario "
                         "(truth, observations, oracle) as npz")
    ap.add_argument("--es", action="store_true",
                    help="compute steady-window energy scores")
    ap.add_argument("--ed", action="store_true",
                    help="compute the steady-window energy distance between "
                         "the particles and the exact filtering distribution")
    ap.add_argument("--out", default=None,
                    help="output directory, one json per run (required)")
    args = ap.parse_args()
    args.reps = list(range(1, args.n_runs + 1))
    if not args.out:
        raise SystemExit("--out required")
    mode_blockwindow(args)


if __name__ == "__main__":
    main()
