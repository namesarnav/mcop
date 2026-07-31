"""Variance reduction techniques for European Monte Carlo pricing.

Plain Monte Carlo converges at O(N^-1/2), so buying one more decimal digit
costs 100x the paths. These estimators attack the numerator instead by
shrinking the variance of the sampled payoff.
"""

from __future__ import annotations

import numpy as np

from mcpricer.payoffs import get_payoff
from mcpricer.pricing import PriceEstimate, estimate_from_discounted_payoffs
from mcpricer.simulation import gbm_terminal_from_normals

__all__ = ["mc_european_price_antithetic", "variance_reduction_pct"]


def mc_european_price_antithetic(
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
) -> PriceEstimate:
    """Price using n_paths/2 draws z paired with their mirrors -z.

    S_T is monotone in z, so the two payoffs in a pair are negatively
    correlated and their average has lower variance than two independent draws.
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be strictly positive")
    if n_paths % 2 != 0:
        raise ValueError("n_paths must be even so that draws can be paired")
    if sigma < 0 or T < 0 or S0 <= 0:
        raise ValueError("invalid market parameters")

    generator = rng if rng is not None else np.random.default_rng(seed)
    payoff = get_payoff(option_type)
    z = generator.standard_normal(n_paths // 2)

    up = payoff(gbm_terminal_from_normals(S0, r, sigma, T, z), K)
    down = payoff(gbm_terminal_from_normals(S0, r, sigma, T, -z), K)
    paired = np.exp(-r * T) * 0.5 * (up + down)
    return estimate_from_discounted_payoffs(paired)


def variance_reduction_pct(se_naive: float, se_reduced: float) -> float:
    """Percentage of variance removed, since variance is the quantity that
    scales with cost: halving the standard error is a 75% reduction."""
    if se_naive <= 0:
        raise ValueError("baseline standard error must be positive")
    return 100.0 * (1.0 - (se_reduced / se_naive) ** 2)
