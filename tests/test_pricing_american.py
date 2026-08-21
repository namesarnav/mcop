import pytest

from mcpricer.binomial import binomial_american_put_price
from mcpricer.black_scholes import bs_call_price, bs_put_price
from mcpricer.pricing import early_exercise_premium, lsm_american_price

BASE = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
N, STEPS = 200_000, 50


def test_lsm_put_matches_binomial_tree_reference():
    lsm = lsm_american_price(**BASE, n_paths=N, n_steps=STEPS, seed=0)
    tree = binomial_american_put_price(**BASE, n_steps=4000)
    assert abs(lsm.price - tree) < 0.05


def test_american_put_is_worth_at_least_the_european_put():
    lsm = lsm_american_price(**BASE, n_paths=N, n_steps=STEPS, seed=1)
    assert lsm.price > bs_put_price(**BASE)


def test_american_call_without_dividends_equals_european_call():
    lsm = lsm_american_price(
        **BASE, n_paths=N, n_steps=STEPS, option_type="call", seed=2
    )
    assert abs(lsm.price - bs_call_price(**BASE)) < 3 * lsm.standard_error


def test_early_exercise_premium_is_positive_for_a_put():
    premium = early_exercise_premium(**BASE, n_paths=N, n_steps=STEPS, seed=3)
    tree_premium = binomial_american_put_price(**BASE, n_steps=4000) - bs_put_price(**BASE)
    assert premium > 0
    assert abs(premium - tree_premium) < 0.1


def test_deep_itm_american_put_approaches_intrinsic_from_below():
    # Exercise is only allowed on the discrete grid, so LSM is a lower bound
    # that tightens towards the intrinsic value 60 as the grid is refined.
    params = dict(S0=40.0, K=100.0, r=0.05, sigma=0.2, T=1.0, n_paths=50_000, seed=4)
    coarse = lsm_american_price(**params, n_steps=25).price
    fine = lsm_american_price(**params, n_steps=100).price
    assert coarse < fine <= 60.0 + 1e-9
    assert abs(fine - 60.0) < 0.1


def test_lsm_is_reproducible_under_a_fixed_seed():
    a = lsm_american_price(**BASE, n_paths=20_000, n_steps=STEPS, seed=5)
    b = lsm_american_price(**BASE, n_paths=20_000, n_steps=STEPS, seed=5)
    assert a.price == b.price


@pytest.mark.parametrize("bad", [dict(n_steps=0), dict(degree=0)])
def test_invalid_lsm_inputs_raise(bad):
    params = dict(BASE, n_paths=1000, n_steps=STEPS)
    params.update(bad)
    with pytest.raises(ValueError):
        lsm_american_price(**params)
