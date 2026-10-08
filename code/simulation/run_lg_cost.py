"""Run the replications of one (mult, N, method) case of the Figure 4 study.

Runs the model of `lg_cost_model.py` and records the RMSE, the energy score,
and the energy distance, one json per replication. The cost axis keeps the
recorded times of `computing_times.npz`, so the run needs no exclusive-timing
conditions and can be oversubscribed freely.
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mult", type=int, required=True,
                    help="propagation sub-step multiplier")
    ap.add_argument("--N", type=int, required=True,
                    help="number of particles")
    ap.add_argument("--method", required=True,
                    help="resampler name (a key of lg_cost_model.RESAMPLERS)")
    ap.add_argument("--n-runs", dest="n_runs", type=int, default=20,
                    help="number of replications, run as 1..n")
    ap.add_argument("--out", required=True,
                    help="output directory, one json per run")
    ap.add_argument("--cuda", default="0",
                    help="CUDA device ids for CUDA_VISIBLE_DEVICES")
    args = ap.parse_args()

    # lg_cost_model.py reads its configuration from argv at import time; feed
    # it the frozen Figure 4 configuration (defaults d=50, kappa=50, T=1000,
    # R=3, cv=2) and the requested multiplier.
    sys.argv = ["lg_cost_model.py", "--substep", str(args.mult),
                "--cuda", args.cuda, "--rep", "1"]
    import numpy as np
    import torch
    import lg_cost_model as lgc
    from gauss_ed import build_terms_sequence, energy_distance

    resamplers = lgc.RESAMPLERS
    if args.method not in resamplers:
        raise SystemExit(f"unknown method {args.method}")
    if args.method == "WDR":
        lgc._ensure_gpu()

    _, Y_obs = lgc.simulate_lg_data(lgc.T_steps, lgc.d, lgc.A_MAT, lgc.Q_PROC,
                                   lgc.H_OBS, lgc.R_NOISE, seed=12345)
    mu0 = np.zeros(lgc.d)
    Sigma0 = lgc.Q_PROC.copy()
    mu_kf, Sigma_kf = lgc.kalman_filter(Y_obs, lgc.A_MAT, lgc.Q_PROC,
                                        lgc.H_OBS, lgc.R_NOISE, mu0, Sigma0)

    # The Gaussian terms of the energy distance depend only on Sigma_t, which
    # is data-independent and converges, so they are built once per case; the
    # per-step evaluation runs in float64 on the GPU when one is available.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    terms, _ = build_terms_sequence(Sigma_kf, device=str(dev))
    mu_dev = torch.as_tensor(mu_kf, dtype=torch.float64, device=dev)

    os.makedirs(args.out, exist_ok=True)
    for rep in range(1, args.n_runs + 1):
        path = os.path.join(args.out,
                            f"{args.method}_mult{args.mult}_N{args.N}_rep{rep}.json")
        if os.path.exists(path):
            print("[skip]", path, flush=True)
            continue

        pf_seed = 24680 + rep * 1000 + args.N           # frozen seed rule
        ed_t = np.full(lgc.T_steps, np.nan)

        def pre_hook(t, Xf, w):
            xt = torch.as_tensor(Xf, dtype=torch.float64, device=dev)
            wt = torch.as_tensor(w, dtype=torch.float64, device=dev)
            ed_t[t] = energy_distance(xt, wt / wt.sum(), mu_dev[t], terms[t])

        X_est, X_samp, w_samp, t_prop, t_resamp, calls = lgc.particle_filter_timed(
            Y_obs, args.N, lgc.R, resamplers[args.method], pf_seed, args.mult,
            pre_hook=pre_hook)

        rmse = float(np.mean(lgc.rmse_over_time(X_est, mu_kf)))
        es_t = lgc.compute_es(X_samp, w_samp, mu_kf)
        # NaN marks a timestep whose weights sit on one particle (distance undefined)
        rec = dict(mult=args.mult, N=args.N, resampler=args.method, rep=rep,
                   pf_seed=pf_seed, rmse=rmse,
                   es=float(np.mean(es_t)),
                   ed=float(np.nanmean(ed_t)),
                   resamp_calls=int(calls),
                   finite=bool(np.isfinite(X_est).all()))
        json.dump(rec, open(path, "w"), indent=1)
        print(f"{args.method} mult={args.mult} N={args.N} rep={rep}: "
              f"rmse={rmse:.6f} es={rec['es']:.6f} ed={rec['ed']:.6f}",
              flush=True)


if __name__ == "__main__":
    main()
