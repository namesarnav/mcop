"""Cox-Ross-Rubinstein binomial tree.

Deterministic reference for American options, where no closed form exists.
Backward induction over a recombining tree with u = e^{sigma sqrt(dt)},
d = 1/u and risk-neutral probability p = (e^{r dt} - d) / (u - d).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "binomial_price",
    "binomial_american_call_price",
    "binomial_american_put_price",
    "binomial_european_call_price",
    "binomial_european_put_price",
]


def binomial_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    option_type: str = "call",
    exercise: str = "european",
) -> float:
    option_type = option_type.strip().lower()
    exercise = exercise.strip().lower()
    if option_type not in {"call", "put"}:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if exercise not in {"european", "american"}:
        raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")
    if n_steps <= 0:
        raise ValueError("n_steps must be strictly positive")
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be strictly positive")

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    discount = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError("risk-neutral probability outside [0, 1]; reduce the step size")

    ups = np.arange(n_steps + 1)
    spot = S0 * u**ups * d ** (n_steps - ups)
    values = np.maximum(spot - K, 0.0) if option_type == "call" else np.maximum(K - spot, 0.0)

    for step in range(n_steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if exercise == "american":
            ups = np.arange(step + 1)
            spot = S0 * u**ups * d ** (step - ups)
            intrinsic = (
                np.maximum(spot - K, 0.0)
                if option_type == "call"
                else np.maximum(K - spot, 0.0)
            )
            values = np.maximum(values, intrinsic)
    return float(values[0])


def binomial_european_call_price(S0, K, r, sigma, T, n_steps: int) -> float:
    return binomial_price(S0, K, r, sigma, T, n_steps, "call", "european")


def binomial_european_put_price(S0, K, r, sigma, T, n_steps: int) -> float:
    return binomial_price(S0, K, r, sigma, T, n_steps, "put", "european")


def binomial_american_call_price(S0, K, r, sigma, T, n_steps: int) -> float:
    return binomial_price(S0, K, r, sigma, T, n_steps, "call", "american")


def binomial_american_put_price(S0, K, r, sigma, T, n_steps: int) -> float:
    return binomial_price(S0, K, r, sigma, T, n_steps, "put", "american")
