import pytest

from mcpricer.binomial import (
    binomial_american_call_price,
    binomial_american_put_price,
    binomial_european_call_price,
    binomial_european_put_price,
)
from mcpricer.black_scholes import bs_call_price, bs_put_price

BASE = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)


def test_european_call_converges_to_black_scholes():
    assert abs(binomial_european_call_price(**BASE, n_steps=5000) - bs_call_price(**BASE)) < 1e-3


def test_european_put_converges_to_black_scholes():
    assert abs(binomial_european_put_price(**BASE, n_steps=5000) - bs_put_price(**BASE)) < 1e-3


def test_european_error_shrinks_with_step_count():
    truth = bs_call_price(**BASE)
    coarse = abs(binomial_european_call_price(**BASE, n_steps=50) - truth)
    fine = abs(binomial_european_call_price(**BASE, n_steps=2000) - truth)
    assert fine < coarse


def test_american_call_equals_european_call_without_dividends():
    # Early exercise throws away the time value of the strike, so it is never
    # optimal for a call on a non-dividend-paying stock.
    american = binomial_american_call_price(**BASE, n_steps=1000)
    european = binomial_european_call_price(**BASE, n_steps=1000)
    assert abs(american - european) < 1e-12


def test_american_put_is_worth_more_than_european_put():
    american = binomial_american_put_price(**BASE, n_steps=1000)
    european = binomial_european_put_price(**BASE, n_steps=1000)
    assert american > european


def test_american_put_price_is_stable_under_refinement():
    coarse = binomial_american_put_price(**BASE, n_steps=1000)
    fine = binomial_american_put_price(**BASE, n_steps=4000)
    assert abs(coarse - fine) < 1e-3


def test_deep_itm_american_put_is_worth_intrinsic_value():
    price = binomial_american_put_price(S0=1.0, K=100.0, r=0.05, sigma=0.2, T=1.0, n_steps=500)
    assert abs(price - 99.0) < 1e-6


def test_american_price_never_below_intrinsic():
    price = binomial_american_put_price(S0=80.0, K=100.0, r=0.05, sigma=0.2, T=1.0, n_steps=500)
    assert price >= 20.0


@pytest.mark.parametrize("bad", [dict(n_steps=0), dict(sigma=0.0), dict(T=0.0)])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        binomial_american_put_price(**dict(BASE, n_steps=100, **bad))


def test_unknown_exercise_style_raises():
    from mcpricer.binomial import binomial_price

    with pytest.raises(ValueError):
        binomial_price(**BASE, n_steps=100, exercise="bermudan")
