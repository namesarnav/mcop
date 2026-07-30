import pytest

from mcpricer.black_scholes import bs_call_price, bs_put_price
from mcpricer.pricing import mc_european_call_price
from mcpricer.variance_reduction import (
    mc_european_price_antithetic,
    variance_reduction_pct,
)

BASE = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
N = 200_000


def test_antithetic_standard_error_lower_at_matched_path_count():
    _, se_naive = mc_european_call_price(**BASE, n_paths=N, seed=3)
    _, se_anti = mc_european_price_antithetic(**BASE, n_paths=N, seed=3)
    assert se_anti < se_naive


def test_antithetic_call_price_is_unbiased():
    estimate = mc_european_price_antithetic(**BASE, n_paths=N, seed=4)
    assert abs(estimate.price - bs_call_price(**BASE)) < 3 * estimate.standard_error


def test_antithetic_put_price_is_unbiased():
    estimate = mc_european_price_antithetic(**BASE, n_paths=N, option_type="put", seed=5)
    assert abs(estimate.price - bs_put_price(**BASE)) < 3 * estimate.standard_error


def test_antithetic_rejects_odd_path_count():
    with pytest.raises(ValueError):
        mc_european_price_antithetic(**BASE, n_paths=1001, seed=0)


def test_variance_reduction_pct_is_computed_on_variance_not_error():
    # Halving the standard error is a 75% variance reduction.
    assert abs(variance_reduction_pct(0.02, 0.01) - 75.0) < 1e-9
    assert variance_reduction_pct(0.02, 0.02) == 0.0
