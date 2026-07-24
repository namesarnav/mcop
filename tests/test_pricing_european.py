import numpy as np
import pytest
from scipy.stats import norm

from mcpricer.black_scholes import bs_call_price, bs_put_price
from mcpricer.payoffs import call_payoff, put_payoff
from mcpricer.pricing import mc_european_call_price, mc_european_put_price

BASE = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)


def test_call_payoff():
    assert call_payoff(np.array([90.0, 100.0, 110.0]), K=100.0).tolist() == [0, 0, 10]


def test_put_payoff():
    assert put_payoff(np.array([90.0, 100.0, 110.0]), K=100.0).tolist() == [10, 0, 0]


def test_payoffs_are_non_negative():
    prices = np.linspace(1.0, 500.0, 1000)
    assert np.all(call_payoff(prices, K=100.0) >= 0)
    assert np.all(put_payoff(prices, K=100.0) >= 0)


def test_payoff_parity():
    # max(S-K,0) - max(K-S,0) = S - K
    prices = np.linspace(1.0, 500.0, 1000)
    diff = call_payoff(prices, K=100.0) - put_payoff(prices, K=100.0)
    assert np.allclose(diff, prices - 100.0)


def test_mc_call_price_matches_black_scholes_within_error_bars():
    estimate = mc_european_call_price(**BASE, n_paths=1_000_000, seed=0)
    assert abs(estimate.price - bs_call_price(**BASE)) < 3 * estimate.standard_error


def test_mc_put_price_matches_black_scholes_within_error_bars():
    estimate = mc_european_put_price(**BASE, n_paths=1_000_000, seed=0)
    assert abs(estimate.price - bs_put_price(**BASE)) < 3 * estimate.standard_error


def test_result_unpacks_as_price_and_standard_error():
    price, se = mc_european_call_price(**BASE, n_paths=10_000, seed=0)
    assert price > 0 and se > 0


def test_mc_price_repeatable_with_fixed_seed():
    p1, _ = mc_european_call_price(**BASE, n_paths=10_000, seed=1)
    p2, _ = mc_european_call_price(**BASE, n_paths=10_000, seed=1)
    assert p1 == p2


def test_zero_volatility_price_is_exact():
    params = dict(BASE, sigma=0.0)
    estimate = mc_european_call_price(**params, n_paths=1000, seed=0)
    assert abs(estimate.price - bs_call_price(**params)) < 1e-10
    assert estimate.standard_error < 1e-12


def test_standard_error_halves_when_paths_quadruple():
    _, se_small = mc_european_call_price(**BASE, n_paths=10_000, seed=2)
    _, se_large = mc_european_call_price(**BASE, n_paths=40_000, seed=2)
    assert 1.7 < se_small / se_large < 2.3


def test_confidence_interval_brackets_the_estimate():
    estimate = mc_european_call_price(**BASE, n_paths=100_000, seed=3)
    low, high = estimate.confidence_interval(0.95)
    assert low < estimate.price < high
    z = norm.ppf(0.975)
    assert abs((high - low) - 2 * z * estimate.standard_error) < 1e-12


def test_statistical_coverage_across_seeds():
    # The 95% CI should contain the true price in roughly 95 of 100 runs.
    truth = bs_call_price(**BASE)
    hits = 0
    for seed in range(100):
        estimate = mc_european_call_price(**BASE, n_paths=20_000, seed=seed)
        low, high = estimate.confidence_interval(0.95)
        hits += low <= truth <= high
    assert 88 <= hits <= 100


def test_invalid_path_count_raises():
    with pytest.raises(ValueError):
        mc_european_call_price(**BASE, n_paths=0, seed=0)
