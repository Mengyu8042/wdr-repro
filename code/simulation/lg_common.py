import numpy as np
from typing import List, Tuple


def build_lg_model(
    d_state: int,
    alpha: float = 0.95,
    q_scale: float = 0.5,
    r_scale: float = 0.1,
    kappa: float = 1.0,
    kappa_seed: int = 2024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a linear Gaussian state-space model.

    When kappa > 1, the process noise covariance is anisotropic:
        Q = q_scale * Q0 diag(lambda_1, ..., lambda_d) Q0^T
    where lambda_j = kappa^{-(j-1)/(d-1)} and Q0 is a random orthogonal matrix
    (from QR of a random Gaussian matrix, seeded for reproducibility).
    This induces correlated state noise with condition number kappa,
    making the intrinsic dimension smaller than d.

    When kappa == 1, Q = q_scale * I_d (isotropic).
    """
    A_mat = alpha * np.eye(d_state)

    if kappa == 1.0 or d_state == 1:
        q_proc = q_scale * np.eye(d_state)
    else:
        rng_q = np.random.default_rng(kappa_seed)
        # Random orthogonal matrix via QR decomposition
        G = rng_q.standard_normal((d_state, d_state))
        Q0, _ = np.linalg.qr(G)
        # Eigenvalues decay geometrically from 1 to 1/kappa
        lambdas = np.array([kappa ** (-(j) / (d_state - 1)) for j in range(d_state)])
        q_proc = q_scale * (Q0 @ np.diag(lambdas) @ Q0.T)

    h_obs = np.eye(d_state)
    r_noise = r_scale * np.eye(d_state)

    return A_mat, q_proc, h_obs, r_noise


def kalman_filter(
    Y_obs: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    R_obs: np.ndarray,
    mu0: np.ndarray,
    Sigma0: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Kalman filter with the LG experiment convention:
    the latent state at t=0 is obtained after one transition from the prior,
    so each step performs predict -> update against Y_obs[t].
    """
    T = Y_obs.shape[0]
    d_state = A.shape[0]

    mu_filt = np.zeros((T, d_state))
    Sigma_filt: List[np.ndarray] = []

    mu = mu0.copy()
    Sigma = Sigma0.copy()

    for t in range(T):
        mu_pred = A @ mu
        Sigma_pred = A @ Sigma @ A.T + Q

        y = Y_obs[t]
        innov = y - H @ mu_pred
        S = H @ Sigma_pred @ H.T + R_obs
        K = Sigma_pred @ H.T @ np.linalg.inv(S)
        mu = mu_pred + K @ innov
        Sigma = (np.eye(d_state) - K @ H) @ Sigma_pred

        mu_filt[t] = mu
        Sigma_filt.append(Sigma.copy())

    return mu_filt, Sigma_filt


def simulate_lg_data(
    T: int,
    d_state: int,
    A: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    R_obs: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    d_obs = H.shape[0]
    X_true = np.zeros((T, d_state))
    x = rng.multivariate_normal(mean=np.zeros(d_state), cov=Q)

    for t in range(T):
        x = A @ x + rng.multivariate_normal(mean=np.zeros(d_state), cov=Q)
        X_true[t] = x

    Y_obs = X_true @ H.T + rng.multivariate_normal(
        mean=np.zeros(d_obs), cov=R_obs, size=T
    )
    return X_true, Y_obs


def propagate_lg_particles(
    X: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
    rng: np.random.Generator,
    n_descendants: int = 1,
) -> np.ndarray:
    N_anc = X.shape[0]
    X_prop = np.repeat(X, n_descendants, axis=0)
    X_prop = X_prop @ A.T
    noise = rng.multivariate_normal(
        mean=np.zeros(A.shape[0]),
        cov=Q,
        size=N_anc * n_descendants,
    )
    return X_prop + noise
