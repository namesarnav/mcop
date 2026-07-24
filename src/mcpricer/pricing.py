"""Monte Carlo pricing engines for European options.

The estimator is the discounted sample mean of the payoff,

    C_hat = e^{-rT} (1/N) sum_i payoff(S_T^i)

with standard error e^{-rT} sigma_payoff / sqrt(N). Prices are always returned
alongside that error: a Monte Carlo price without one is not a result.
"""

from __future__ import annotations

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
