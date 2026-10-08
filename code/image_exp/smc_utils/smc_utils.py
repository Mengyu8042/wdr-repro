"""SMC utilities for diffusion image experiments.

Three resamplers:
    "systematic" : vanilla TDS systematic resampling.
    "wdr"        : TDS-WDR — latent-space WDR via VAE encoding.
    "t-wdr"      : TDS-T-WDR — latent-space T-WDR.
"""
import numpy as np
import torch as th
from numba import jit
from numpy import random
from .image_ot_resampling import resample_wdr, resample_t_wdr


# Cache for VAE checkpoints to avoid repeated loading.
_VAE_CACHE = {}


def get_cached_vae(path, device):
    key = (path, str(device))
    if key not in _VAE_CACHE:
        # Imported lazily so the release tree only loads vae.py when wdr / t-wdr
        # are actually selected. Run from inside `image_exp/` so this resolves.
        from models.vae import load_vae
        print(f"Loading VAE from {path} to {device}...")
        vae = load_vae(path=path, device=device)
        vae.eval()
        _VAE_CACHE[key] = vae
    return _VAE_CACHE[key]


def compute_ess(w, dim=0):
    ess = (w.sum(dim=dim)) ** 2 / th.sum(w ** 2, dim=dim)
    return ess


def compute_ess_from_log_w(log_w, dim=0):
    return compute_ess(normalize_weights(log_w, dim=dim), dim=dim)


def normalize_weights(log_weights, dim=0):
    return th.exp(normalize_log_weights(log_weights, dim=dim))


def normalize_log_weights(log_weights, dim):
    log_weights = log_weights - log_weights.max(dim=dim, keepdims=True)[0]
    log_weights = log_weights - th.logsumexp(log_weights, dim=dim, keepdims=True)
    return log_weights


@jit(nopython=True)
def inverse_cdf(su, W):
    """Inverse CDF for a finite distribution.

    Args
        su: (M,) ndarray of M ordered uniform variates in [0, 1].
        W: (N,) ndarray of N normalized weights.

    Returns
        (M,) ndarray of indices in [0, N - 1].
    """
    j = 0
    s = W[0]
    M = su.shape[0]
    A = np.empty(M, dtype=np.int64)
    for n in range(M):
        while su[n] > s:
            if j == M - 1:
                break
            j += 1
            s += W[j]
        A[n] = j
    return A


def systematic(W, X, M, x0_pred):
    """Systematic resampling."""
    su = (random.rand(1) + np.arange(M)) / M
    idx = inverse_cdf(su, W)
    return idx, X[idx]


@th.no_grad()
def wdr_latent(
    W: np.ndarray,
    X: th.Tensor,
    M: int,
    x0_pred,
    flatten=True,
    vae_path="./models/checkpoints/vae_mnist_z32.pt",
):
    """WDR in the VAE latent space.

    Encodes the predicted x_0 with the dataset's VAE, runs WDR on the latent
    codes, then maps back to image space via the resampling indices.
    """
    device = X.device
    dtype = X.dtype
    P = X.shape[0]
    orig_shape = tuple(X.shape)

    vae = get_cached_vae(path=vae_path, device=device)

    # Map predicted x_0 to [0, 1] before encoding.
    x0 = x0_pred.to(device=device, dtype=dtype)
    x0 = x0.clamp(-1, 1)
    x_in = ((x0 + 1.0) / 2.0).clamp(0, 1)

    Z = vae.encode(x_in)[0].detach()
    if flatten and Z.ndim > 2:
        Z = Z.view(P, -1)

    rng = np.random.default_rng(2025)

    _, resample_idx = resample_wdr(
        particles=Z,
        weights=W,
        rng=rng,
        new_size=M,
    )

    resample_idx_torch = th.from_numpy(resample_idx).long().to(X.device)
    Y = X[resample_idx_torch]

    if flatten and len(orig_shape) > 2:
        Y = Y.view((Y.shape[0],) + orig_shape[1:])

    return resample_idx, Y


@th.no_grad()
def t_wdr_latent(
    W: np.ndarray,
    X: th.Tensor,
    M: int,
    x0_pred,
    flatten=True,
    vae_path="./models/checkpoints/vae_mnist_z32.pt",
):
    """T-WDR in the VAE latent space.

    When pred_xstart is None at t = 0, fall back to encoding the current
    particles X for the latent feature.
    """
    device = X.device
    dtype = X.dtype
    P = X.shape[0]
    orig_shape = tuple(X.shape)

    vae = get_cached_vae(path=vae_path, device=device)

    x0_src = X if x0_pred is None else x0_pred
    x0 = x0_src.to(device=device, dtype=dtype)
    x0 = x0.clamp(-1, 1)
    x_in = ((x0 + 1.0) / 2.0).clamp(0, 1)

    Z = vae.encode(x_in)[0].detach()
    if flatten and Z.ndim > 2:
        Z = Z.view(P, -1)

    rng = np.random.default_rng(2025)

    _, new_weights, resample_idx = resample_t_wdr(
        particles=Z,
        weights=W,
        rng=rng,
        new_size=M,
    )

    resample_idx_torch = th.from_numpy(resample_idx).long().to(X.device)
    Y = X[resample_idx_torch]

    if flatten and len(orig_shape) > 2:
        Y = Y.view((Y.shape[0],) + orig_shape[1:])

    return resample_idx, Y, new_weights


Resample_dict = {
    "systematic": systematic,
    "wdr": wdr_latent,
    "t-wdr": t_wdr_latent,
}


_LATENT_STRATEGIES = {"wdr", "t-wdr"}


def resampling_function(resample_strategy="systematic", ess_threshold=None,
                        verbose=False, vae_path=None):
    """Build a resample callback compatible with smc_FK."""
    resample_fn = Resample_dict[resample_strategy]

    def resample(log_w, xt, x0_pred, target_P=None):
        normalized_weights = normalize_weights(log_w, dim=-1)
        P = log_w.shape[0]
        ess = compute_ess(normalized_weights, dim=0)

        condition_ess = (ess_threshold is None) or (ess < P * ess_threshold)
        condition_count = (target_P is not None) and (target_P != P)

        if condition_ess or condition_count:
            if verbose:
                print("resample")

            kwargs = {}
            if resample_strategy in _LATENT_STRATEGIES and vae_path is not None:
                kwargs['vae_path'] = vae_path

            M_val = target_P if target_P is not None else P

            ret = resample_fn(W=np.array(normalized_weights.cpu()), X=xt, M=M_val,
                              x0_pred=x0_pred, **kwargs)
            resample_indices = th.from_numpy(ret[0])
            xt_new = ret[1]

            if len(ret) > 2:
                new_weights_np = ret[2]
                new_log_weights = th.from_numpy(np.log(new_weights_np) + np.log(P)).to(log_w)
            else:
                new_log_weights = None

            is_resampled = True
        else:
            resample_indices = th.arange(P)
            is_resampled = False
            xt_new = xt
            new_log_weights = None

        return resample_indices, xt_new, is_resampled, new_log_weights

    return resample
