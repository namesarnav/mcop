"""Risk-neutral asset path simulation.

GBM is simulated in log space, where the SDE has the exact solution

    S_T = S0 exp[(r - sigma^2/2) T + sigma sqrt(T) Z],   Z ~ N(0, 1)

so time-stepping introduces no discretisation error and prices cannot go
negative.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["simulate_gbm_terminal", "simulate_gbm_paths"]


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
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    return S0 * np.exp(drift + diffusion)


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
