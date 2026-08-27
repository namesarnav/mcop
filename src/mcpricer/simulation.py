"""Risk-neutral asset path simulation.

GBM is simulated in log space, where the SDE has the exact solution

    S_T = S0 exp[(r - sigma^2/2) T + sigma sqrt(T) Z],   Z ~ N(0, 1)

so time-stepping introduces no discretisation error and prices cannot go
negative.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "gbm_terminal_from_normals",
    "simulate_gbm_terminal",
    "simulate_gbm_paths",
    "simulate_merton_terminal",
    "simulate_heston_paths",
]


def _make_rng(seed: int | None, rng: np.random.Generator | None) -> np.random.Generator:
    if rng is not None and seed is not None:
        raise ValueError("pass either seed or rng, not both")
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _validate(S0: float, sigma: float, T: float, n_paths: int) -> None:
    if S0 <= 0:
        raise ValueError("spot price S0 must be strictly positive")
    if sigma < 0:
        raise ValueError("volatility sigma must be non-negative")
    if T < 0:
        raise ValueError("time to maturity T must be non-negative")
    if n_paths <= 0:
        raise ValueError("n_paths must be strictly positive")


def gbm_terminal_from_normals(
    S0: float, r: float, sigma: float, T: float, z: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Map standard normal draws to terminal prices.

    Separated out so callers can supply their own draws: antithetic variates
    reuse -z, and finite-difference Greeks reuse the same z across both legs.
    """
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def simulate_gbm_terminal(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Sample S_T directly, one normal draw per path.

    European payoffs depend only on the terminal price, so there is no reason
    to walk the intermediate grid.
    """
    _validate(S0, sigma, T, n_paths)
    generator = _make_rng(seed, rng)
    z = generator.standard_normal(n_paths)
    return gbm_terminal_from_normals(S0, r, sigma, T, z)


def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Full time-stepped paths, shape (n_paths, n_steps + 1) including S0.

    Needed for path-dependent payoffs and early exercise.
    """
    _validate(S0, sigma, T, n_paths)
    if n_steps <= 0:
        raise ValueError("n_steps must be strictly positive")
    generator = _make_rng(seed, rng)
    dt = T / n_steps
    z = generator.standard_normal((n_paths, n_steps))
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


def simulate_merton_terminal(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    *,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Merton jump-diffusion terminal prices.

        S_T = S0 exp[(r - lambda k - sigma^2/2) T + sigma sqrt(T) Z + sum_i Y_i]

    with N_T ~ Poisson(lambda T) log-jumps Y_i ~ N(jump_mean, jump_std^2) and
    k = e^{jump_mean + jump_std^2/2} - 1. The -lambda k drift offset is the
    compensator that keeps the discounted price a martingale.

    Conditional on N_T the jump sum is itself normal, so no per-jump loop is
    needed.
    """
    _validate(S0, sigma, T, n_paths)
    if jump_intensity < 0:
        raise ValueError("jump_intensity must be non-negative")
    if jump_std < 0:
        raise ValueError("jump_std must be non-negative")

    generator = _make_rng(seed, rng)
    k = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    n_jumps = generator.poisson(jump_intensity * T, n_paths)
    z_diffusion = generator.standard_normal(n_paths)
    z_jump = generator.standard_normal(n_paths)

    jump_component = jump_mean * n_jumps + jump_std * np.sqrt(n_jumps) * z_jump
    drift = (r - jump_intensity * k - 0.5 * sigma**2) * T
    return S0 * np.exp(drift + sigma * np.sqrt(T) * z_diffusion + jump_component)


def simulate_heston_paths(
    S0: float,
    r: float,
    T: float,
    n_paths: int,
    n_steps: int,
    *,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    return_variance: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Heston stochastic volatility, full truncation Euler scheme.

        dS = r S dt + sqrt(v) S dW1
        dv = kappa (theta - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho

    Euler steps can drive v below zero even when the Feller condition holds, so
    every use of v is replaced by max(v, 0). Full truncation keeps the negative
    value in the state and only truncates where v is read, which is the variant
    with the smallest discretisation bias.
    """
    _validate(S0, 0.0, T, n_paths)
    if n_steps <= 0:
        raise ValueError("n_steps must be strictly positive")
    if v0 < 0 or theta < 0 or xi < 0 or kappa < 0:
        raise ValueError("v0, kappa, theta and xi must be non-negative")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")

    generator = _make_rng(seed, rng)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    z1 = generator.standard_normal((n_paths, n_steps))
    z2 = rho * z1 + np.sqrt(1.0 - rho**2) * generator.standard_normal((n_paths, n_steps))

    log_s = np.empty((n_paths, n_steps + 1))
    variance = np.empty((n_paths, n_steps + 1))
    log_s[:, 0] = np.log(S0)
    variance[:, 0] = v0

    for step in range(n_steps):
        v_plus = np.maximum(variance[:, step], 0.0)
        vol = np.sqrt(v_plus)
        log_s[:, step + 1] = (
            log_s[:, step] + (r - 0.5 * v_plus) * dt + vol * sqrt_dt * z1[:, step]
        )
        variance[:, step + 1] = (
            variance[:, step]
            + kappa * (theta - v_plus) * dt
            + xi * vol * sqrt_dt * z2[:, step]
        )

    paths = np.exp(log_s)
    return (paths, variance) if return_variance else paths
