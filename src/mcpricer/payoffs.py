"""Vanilla option payoff functions."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["call_payoff", "put_payoff", "get_payoff"]


def call_payoff(S: ArrayLike, K: float) -> NDArray[np.float64]:
    """max(S - K, 0)."""
    return np.maximum(np.asarray(S, dtype=np.float64) - K, 0.0)


def put_payoff(S: ArrayLike, K: float) -> NDArray[np.float64]:
    """max(K - S, 0)."""
    return np.maximum(K - np.asarray(S, dtype=np.float64), 0.0)


def get_payoff(option_type: str) -> Callable[[ArrayLike, float], NDArray[np.float64]]:
    """Look up a payoff function by option type."""
    normalised = option_type.strip().lower()
    if normalised == "call":
        return call_payoff
    if normalised == "put":
        return put_payoff
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
