import numpy as np
import pytest

from mcpricer.black_scholes import bs_call_delta, bs_put_delta
from mcpricer.greeks import mc_delta_finite_difference

BASE = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
N = 500_000


def test_finite_difference_delta_matches_bs_delta():
    estimate = mc_delta_finite_difference(**BASE, n_paths=N, bump=0.01, seed=0)
    assert abs(estimate.value - bs_call_delta(**BASE)) < 3 * estimate.standard_error


def test_finite_difference_put_delta_matches_bs_delta():
    estimate = mc_delta_finite_difference(
        **BASE, n_paths=N, bump=0.01, option_type="put", seed=1
    )
    assert abs(estimate.value - bs_put_delta(**BASE)) < 3 * estimate.standard_error


def test_common_random_numbers_are_required_for_a_usable_estimate():
    # Both legs must share draws. With independent draws the difference of two
    # noisy prices is divided by a tiny bump, so the noise is amplified enormously.
    crn = mc_delta_finite_difference(**BASE, n_paths=N, bump=0.01, seed=2)
    independent = mc_delta_finite_difference(
        **BASE, n_paths=N, bump=0.01, seed=2, common_random_numbers=False
    )
    assert crn.standard_error < independent.standard_error / 100


def test_delta_estimate_is_insensitive_to_bump_size_under_crn():
    truth = bs_call_delta(**BASE)
    for bump in (0.001, 0.01, 0.1, 1.0):
        estimate = mc_delta_finite_difference(**BASE, n_paths=N, bump=bump, seed=3)
        assert abs(estimate.value - truth) < 0.01


def test_zero_bump_raises():
    with pytest.raises(ValueError):
        mc_delta_finite_difference(**BASE, n_paths=1000, bump=0.0, seed=0)


def test_finite_difference_delta_is_reproducible():
    a = mc_delta_finite_difference(**BASE, n_paths=10_000, bump=0.01, seed=5)
    b = mc_delta_finite_difference(**BASE, n_paths=10_000, bump=0.01, seed=5)
    assert a.value == b.value
