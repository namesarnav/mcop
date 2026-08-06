import pytest

from mcpricer.black_scholes import bs_call_price, bs_put_price
from mcpricer.pricing import mc_european_call_price, mc_european_put_price
from mcpricer.variance_reduction import (
    mc_european_price_antithetic,
    mc_european_price_control_variate,
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


def test_control_variate_standard_error_lower_at_matched_path_count():
    _, se_naive = mc_european_call_price(**BASE, n_paths=N, seed=6)
    _, se_cv, _ = mc_european_price_control_variate(**BASE, n_paths=N, seed=6)
    assert se_cv < se_naive


def test_control_variate_call_price_is_unbiased():
    estimate = mc_european_price_control_variate(**BASE, n_paths=N, seed=7)
    assert abs(estimate.price - bs_call_price(**BASE)) < 3 * estimate.standard_error


def test_control_variate_put_price_is_unbiased():
    estimate = mc_european_price_control_variate(
        **BASE, n_paths=N, option_type="put", seed=8
    )
    assert abs(estimate.price - bs_put_price(**BASE)) < 3 * estimate.standard_error


def test_control_coefficient_is_positive_for_a_call():
    # A call payoff rises with S_T, so the optimal c = Cov(X, S_T)/Var(S_T) > 0.
    estimate = mc_european_price_control_variate(**BASE, n_paths=N, seed=9)
    assert estimate.coefficient > 0


def test_deep_itm_control_variate_removes_almost_all_variance():
    # Deep in the money the payoff is nearly S_T - K, so S_T is a near-perfect
    # control and the residual variance should collapse.
    params = dict(BASE, S0=300.0)
    _, se_naive = mc_european_call_price(**params, n_paths=N, seed=10)
    _, se_cv, _ = mc_european_price_control_variate(**params, n_paths=N, seed=10)
    assert se_cv < 0.02 * se_naive


def test_variance_reduction_pct_is_computed_on_variance_not_error():
    # Halving the standard error is a 75% variance reduction.
    assert abs(variance_reduction_pct(0.02, 0.01) - 75.0) < 1e-9
    assert variance_reduction_pct(0.02, 0.02) == 0.0


def test_both_techniques_beat_naive_on_the_same_seed():
    _, se_naive = mc_european_call_price(**BASE, n_paths=N, seed=13)
    _, se_anti = mc_european_price_antithetic(**BASE, n_paths=N, seed=13)
    _, se_cv, _ = mc_european_price_control_variate(**BASE, n_paths=N, seed=13)
    assert variance_reduction_pct(se_naive, se_anti) > 0
    assert variance_reduction_pct(se_naive, se_cv) > 0


def test_comparison_table_reports_all_three_estimators():
    from mcpricer.variance_reduction import compare_estimators

    rows = compare_estimators(**BASE, n_paths=N, seed=14)
    assert [row["method"] for row in rows] == ["naive", "antithetic", "control variate"]
    assert rows[0]["variance_reduction_pct"] == 0.0
    assert all(row["variance_reduction_pct"] > 0 for row in rows[1:])
    truth = bs_call_price(**BASE)
    assert all(abs(row["price"] - truth) < 3 * row["standard_error"] for row in rows)
