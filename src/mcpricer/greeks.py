"""Monte Carlo Greeks estimators.

Finite-difference estimators must reuse the same normal draws for both legs of
the bump. Without that, the difference of two independently noisy prices is
divided by a small bump and the noise is amplified by 1/bump, which usually
swamps the signal entirely.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from mcpricer.payoffs import get_payoff
from mcpricer.simulation import gbm_terminal_from_normals

__all__ = ["GreekEstimate", "mc_delta_finite_difference", "mc_delta_pathwise"]


class GreekEstimate(NamedTuple):
    value: float
    standard_error: float


def _estimate(samples: np.ndarray) -> GreekEstimate:
    n = samples.size
    se = float(samples.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return GreekEstimate(float(samples.mean()), se)


def mc_delta_finite_difference(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    option_type: str = "call",
    *,
    bump: float = 0.01,
    common_random_numbers: bool = True,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> GreekEstimate:
    """Central difference [C(S0+h) - C(S0-h)] / 2h under common random numbers.

    Set common_random_numbers=False only to demonstrate why it matters.
    """
    if bump <= 0:
        raise ValueError("bump must be strictly positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be strictly positive")

    generator = rng if rng is not None else np.random.default_rng(seed)
    payoff = get_payoff(option_type)
    discount = np.exp(-r * T)

    z_up = generator.standard_normal(n_paths)
    z_down = z_up if common_random_numbers else generator.standard_normal(n_paths)

    up = payoff(gbm_terminal_from_normals(S0 + bump, r, sigma, T, z_up), K)
    down = payoff(gbm_terminal_from_normals(S0 - bump, r, sigma, T, z_down), K)
    return _estimate(discount * (up - down) / (2.0 * bump))


def mc_delta_pathwise(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    option_type: str = "call",
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> GreekEstimate:
    """Differentiate the payoff along each path instead of bumping.

    S_T is linear in S0, so dS_T/dS0 = S_T/S0 and the estimator is

        call:  e^{-rT} 1{S_T > K} S_T / S0
        put:  -e^{-rT} 1{S_T < K} S_T / S0

    No bump means no bias to trade off against variance. Valid here because the
    vanilla payoff is Lipschitz and differentiable almost everywhere.
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be strictly positive")

    normalised = option_type.strip().lower()
    if normalised not in {"call", "put"}:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    generator = rng if rng is not None else np.random.default_rng(seed)
    z = generator.standard_normal(n_paths)
    terminal = gbm_terminal_from_normals(S0, r, sigma, T, z)
    discount = np.exp(-r * T)

    if normalised == "call":
        samples = discount * (terminal > K) * terminal / S0
    else:
        samples = -discount * (terminal < K) * terminal / S0
    return _estimate(samples)
