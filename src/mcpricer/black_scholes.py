"""Closed-form Black-Scholes-Merton prices and Greeks.

Deterministic reference used to validate the Monte Carlo engine. All functions
broadcast over arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

__all__ = [
    "bs_call_price",
    "bs_put_price",
    "bs_call_delta",
    "bs_put_delta",
    "bs_vega",
    "bs_gamma",
    "bs_theta",
    "bs_rho",
    "bs_price",
]

FloatOrArray = float | NDArray[np.float64]


def _validate(
    S0: ArrayLike, K: ArrayLike, r: ArrayLike, sigma: ArrayLike, T: ArrayLike
) -> tuple[NDArray[np.float64], ...]:
    arrays = tuple(np.asarray(x, dtype=np.float64) for x in (S0, K, r, sigma, T))
    S0_a, K_a, _, sigma_a, T_a = arrays
    if np.any(S0_a <= 0):
        raise ValueError("spot price S0 must be strictly positive")
    if np.any(K_a <= 0):
        raise ValueError("strike K must be strictly positive")
    if np.any(sigma_a < 0):
        raise ValueError("volatility sigma must be non-negative")
    if np.any(T_a < 0):
        raise ValueError("time to maturity T must be non-negative")
    return arrays


def _check_option_type(option_type: str) -> str:
    normalised = option_type.strip().lower()
    if normalised not in {"call", "put"}:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return normalised


def _unwrap(x: NDArray[np.float64]) -> FloatOrArray:
    return float(x) if np.ndim(x) == 0 else x


def _d1_d2(
    S0: NDArray[np.float64],
    K: NDArray[np.float64],
    r: NDArray[np.float64],
    sigma: NDArray[np.float64],
    T: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """d1 = [ln(S0/K) + (r + sigma^2/2)T] / (sigma sqrt(T)), d2 = d1 - sigma sqrt(T).

    The third return value flags sigma*sqrt(T) == 0, where d1 and d2 are
    undefined and the payoff is a deterministic function of the forward.
    """
    total_vol = sigma * np.sqrt(T)
    degenerate = total_vol <= 0.0
    safe_vol = np.where(degenerate, 1.0, total_vol)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / safe_vol
    d2 = d1 - total_vol
    return d1, d2, degenerate


def bs_call_price(
    S0: ArrayLike, K: ArrayLike, r: ArrayLike, sigma: ArrayLike, T: ArrayLike
) -> FloatOrArray:
    """C = S0 N(d1) - K e^{-rT} N(d2)."""
    S0, K, r, sigma, T = _validate(S0, K, r, sigma, T)
    d1, d2, degenerate = _d1_d2(S0, K, r, sigma, T)
    discount = np.exp(-r * T)
    forward = S0 * np.exp(r * T)
    price = np.where(
        degenerate,
        discount * np.maximum(forward - K, 0.0),
        S0 * norm.cdf(d1) - K * discount * norm.cdf(d2),
    )
    return _unwrap(price)


def bs_put_price(
    S0: ArrayLike, K: ArrayLike, r: ArrayLike, sigma: ArrayLike, T: ArrayLike
) -> FloatOrArray:
    """P = K e^{-rT} N(-d2) - S0 N(-d1)."""
    S0, K, r, sigma, T = _validate(S0, K, r, sigma, T)
    d1, d2, degenerate = _d1_d2(S0, K, r, sigma, T)
    discount = np.exp(-r * T)
    forward = S0 * np.exp(r * T)
    price = np.where(
        degenerate,
        discount * np.maximum(K - forward, 0.0),
        K * discount * norm.cdf(-d2) - S0 * norm.cdf(-d1),
    )
    return _unwrap(price)


def bs_call_delta(
    S0: ArrayLike, K: ArrayLike, r: ArrayLike, sigma: ArrayLike, T: ArrayLike
) -> FloatOrArray:
    """dC/dS0 = N(d1)."""
    S0, K, r, sigma, T = _validate(S0, K, r, sigma, T)
    d1, _, degenerate = _d1_d2(S0, K, r, sigma, T)
    forward = S0 * np.exp(r * T)
    delta = np.where(degenerate, (forward > K).astype(np.float64), norm.cdf(d1))
    return _unwrap(delta)


def bs_put_delta(
    S0: ArrayLike, K: ArrayLike, r: ArrayLike, sigma: ArrayLike, T: ArrayLike
) -> FloatOrArray:
    """dP/dS0 = N(d1) - 1."""
    call_delta = np.asarray(bs_call_delta(S0, K, r, sigma, T), dtype=np.float64)
    return _unwrap(call_delta - 1.0)


def bs_vega(
    S0: ArrayLike, K: ArrayLike, r: ArrayLike, sigma: ArrayLike, T: ArrayLike
) -> FloatOrArray:
    """dC/dsigma = S0 phi(d1) sqrt(T). Same for calls and puts."""
    S0, K, r, sigma, T = _validate(S0, K, r, sigma, T)
    d1, _, degenerate = _d1_d2(S0, K, r, sigma, T)
    vega = np.where(degenerate, 0.0, S0 * norm.pdf(d1) * np.sqrt(T))
    return _unwrap(vega)


def bs_gamma(
    S0: ArrayLike, K: ArrayLike, r: ArrayLike, sigma: ArrayLike, T: ArrayLike
) -> FloatOrArray:
    """d2C/dS0^2 = phi(d1) / (S0 sigma sqrt(T)). Same for calls and puts."""
    S0, K, r, sigma, T = _validate(S0, K, r, sigma, T)
    d1, _, degenerate = _d1_d2(S0, K, r, sigma, T)
    total_vol = sigma * np.sqrt(T)
    safe_vol = np.where(degenerate, 1.0, total_vol)
    gamma = np.where(degenerate, 0.0, norm.pdf(d1) / (S0 * safe_vol))
    return _unwrap(gamma)


def bs_theta(
    S0: ArrayLike,
    K: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    T: ArrayLike,
    option_type: str = "call",
) -> FloatOrArray:
    """Decay per year: -S0 phi(d1) sigma / (2 sqrt(T)) -/+ r K e^{-rT} N(+/-d2)."""
    option_type = _check_option_type(option_type)
    S0, K, r, sigma, T = _validate(S0, K, r, sigma, T)
    d1, d2, degenerate = _d1_d2(S0, K, r, sigma, T)
    discount = np.exp(-r * T)
    safe_T = np.where(T <= 0, 1.0, T)
    decay = -S0 * norm.pdf(d1) * sigma / (2.0 * np.sqrt(safe_T))
    if option_type == "call":
        theta = decay - r * K * discount * norm.cdf(d2)
    else:
        theta = decay + r * K * discount * norm.cdf(-d2)
    return _unwrap(np.where(degenerate, 0.0, theta))


def bs_rho(
    S0: ArrayLike,
    K: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    T: ArrayLike,
    option_type: str = "call",
) -> FloatOrArray:
    """dC/dr = K T e^{-rT} N(d2); dP/dr = -K T e^{-rT} N(-d2)."""
    option_type = _check_option_type(option_type)
    S0, K, r, sigma, T = _validate(S0, K, r, sigma, T)
    _, d2, degenerate = _d1_d2(S0, K, r, sigma, T)
    discount = np.exp(-r * T)
    if option_type == "call":
        rho = K * T * discount * norm.cdf(d2)
    else:
        rho = -K * T * discount * norm.cdf(-d2)
    return _unwrap(np.where(degenerate, 0.0, rho))


def bs_price(
    S0: ArrayLike,
    K: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    T: ArrayLike,
    option_type: str = "call",
) -> FloatOrArray:
    """Dispatch to the call or put price."""
    if _check_option_type(option_type) == "call":
        return bs_call_price(S0, K, r, sigma, T)
    return bs_put_price(S0, K, r, sigma, T)
