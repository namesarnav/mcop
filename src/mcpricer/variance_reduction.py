"""Variance reduction techniques for European Monte Carlo pricing.

Plain Monte Carlo converges at O(N^-1/2), so buying one more decimal digit
costs 100x the paths. These estimators attack the numerator instead by
shrinking the variance of the sampled payoff.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from mcpricer.payoffs import get_payoff
from mcpricer.pricing import PriceEstimate, estimate_from_discounted_payoffs
from mcpricer.simulation import gbm_terminal_from_normals, simulate_gbm_terminal

__all__ = [
    "mc_european_price_antithetic",
    "mc_european_price_control_variate",
    "variance_reduction_pct",
]


class ControlVariateEstimate(NamedTuple):
    """Price, standard error and the fitted control coefficient c."""

    price: float
    standard_error: float
    coefficient: float

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        return PriceEstimate(self.price, self.standard_error).confidence_interval(level)


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


def mc_european_price_control_variate(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    option_type: str = "call",
    *,
    pilot_fraction: float = 0.1,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> ControlVariateEstimate:
    """Price using S_T as a control, since E[S_T] = S0 e^{rT} is known exactly.

    The estimator is X - c(Y - E[Y]) with c = Cov(X, Y)/Var(Y), the least
    squares slope of the payoff on the control. c is fitted on a pilot sample
    and applied to a disjoint main sample, so it is independent of the draws it
    corrects and the estimator stays unbiased.
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be strictly positive")
    if not 0.0 < pilot_fraction < 1.0:
        raise ValueError("pilot_fraction must lie in (0, 1)")

    generator = rng if rng is not None else np.random.default_rng(seed)
    payoff = get_payoff(option_type)
    discount = np.exp(-r * T)
    control_mean = S0 * np.exp(r * T)

    n_pilot = min(max(int(n_paths * pilot_fraction), 2), n_paths - 2)
    n_main = n_paths - n_pilot

    pilot = simulate_gbm_terminal(
        S0=S0, r=r, sigma=sigma, T=T, n_paths=n_pilot, rng=generator
    )
    pilot_payoffs = discount * payoff(pilot, K)
    control_variance = pilot.var(ddof=1)
    if control_variance <= 0:
        coefficient = 0.0
    else:
        coefficient = float(np.cov(pilot_payoffs, pilot, ddof=1)[0, 1] / control_variance)

    main = simulate_gbm_terminal(
        S0=S0, r=r, sigma=sigma, T=T, n_paths=n_main, rng=generator
    )
    adjusted = discount * payoff(main, K) - coefficient * (main - control_mean)
    estimate = estimate_from_discounted_payoffs(adjusted)
    return ControlVariateEstimate(estimate.price, estimate.standard_error, coefficient)


def variance_reduction_pct(se_naive: float, se_reduced: float) -> float:
    """Percentage of variance removed, since variance is the quantity that
    scales with cost: halving the standard error is a 75% reduction."""
    if se_naive <= 0:
        raise ValueError("baseline standard error must be positive")
    return 100.0 * (1.0 - (se_reduced / se_naive) ** 2)
