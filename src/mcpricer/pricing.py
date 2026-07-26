"""Monte Carlo pricing engines for European options.

The estimator is the discounted sample mean of the payoff,

    C_hat = e^{-rT} (1/N) sum_i payoff(S_T^i)

with standard error e^{-rT} sigma_payoff / sqrt(N). Prices are always returned
alongside that error: a Monte Carlo price without one is not a result.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy.stats import norm

from mcpricer.payoffs import get_payoff
from mcpricer.simulation import simulate_gbm_terminal

__all__ = [
    "PriceEstimate",
    "mc_european_price",
    "mc_european_call_price",
    "mc_european_put_price",
    "convergence_study",
]


class PriceEstimate(NamedTuple):
    """A Monte Carlo price and its standard error. Unpacks as (price, se)."""

    price: float
    standard_error: float

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        if not 0.0 < level < 1.0:
            raise ValueError("confidence level must lie in (0, 1)")
        z = norm.ppf(0.5 + level / 2.0)
        half_width = z * self.standard_error
        return self.price - half_width, self.price + half_width


def estimate_from_discounted_payoffs(payoffs: np.ndarray) -> PriceEstimate:
    """Sample mean and standard error of already-discounted payoffs."""
    n = payoffs.size
    standard_error = float(payoffs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return PriceEstimate(float(payoffs.mean()), standard_error)


def mc_european_price(
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
    payoff = get_payoff(option_type)
    terminal = simulate_gbm_terminal(
        S0=S0, r=r, sigma=sigma, T=T, n_paths=n_paths, seed=seed, rng=rng
    )
    discounted = np.exp(-r * T) * payoff(terminal, K)
    return estimate_from_discounted_payoffs(discounted)


def mc_european_call_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> PriceEstimate:
    return mc_european_price(
        S0, K, r, sigma, T, n_paths, "call", seed=seed, rng=rng
    )


def mc_european_put_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> PriceEstimate:
    return mc_european_price(
        S0, K, r, sigma, T, n_paths, "put", seed=seed, rng=rng
    )


def convergence_study(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    path_counts: Sequence[int],
    option_type: str = "call",
    *,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Price at each path count in turn, for the convergence plot.

    Each run gets its own generator seeded from a common sequence so the points
    are independent rather than nested subsets of one sample path.
    """
    seeds = np.random.SeedSequence(seed).spawn(len(path_counts))
    prices, errors = [], []
    for n, child in zip(path_counts, seeds):
        estimate = mc_european_price(
            S0, K, r, sigma, T, n, option_type,
            rng=np.random.default_rng(child),
        )
        prices.append(estimate.price)
        errors.append(estimate.standard_error)
    return {
        "n_paths": np.asarray(path_counts, dtype=np.int64),
        "price": np.asarray(prices, dtype=np.float64),
        "standard_error": np.asarray(errors, dtype=np.float64),
    }
