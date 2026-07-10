import numpy as np
import pytest

from mcpricer.black_scholes import (
    bs_call_delta,
    bs_call_price,
    bs_gamma,
    bs_put_delta,
    bs_put_price,
    bs_vega,
)

# Hull's canonical example: call 10.4506, put 5.5735.
BASE = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)


def test_bs_call_price_known_value():
    assert abs(bs_call_price(**BASE) - 10.4506) < 1e-3


def test_bs_put_price_known_value():
    assert abs(bs_put_price(**BASE) - 5.5735) < 1e-3


def test_put_call_parity():
    call = bs_call_price(**BASE)
    put = bs_put_price(**BASE)
    forward = BASE["S0"] - BASE["K"] * np.exp(-BASE["r"] * BASE["T"])
    assert abs((call - put) - forward) < 1e-10


def test_bs_price_zero_volatility():
    params = dict(BASE, sigma=0.0)
    expected = max(100.0 * np.exp(0.05) - 100.0, 0.0) * np.exp(-0.05)
    assert abs(bs_call_price(**params) - expected) < 1e-12


def test_bs_price_zero_time_to_maturity_is_intrinsic():
    assert abs(bs_call_price(**dict(BASE, T=0.0, S0=120.0)) - 20.0) < 1e-12
    assert abs(bs_call_price(**dict(BASE, T=0.0, S0=80.0)) - 0.0) < 1e-12
    assert abs(bs_put_price(**dict(BASE, T=0.0, S0=80.0)) - 20.0) < 1e-12


def test_bs_deep_itm_call_converges_to_intrinsic():
    params = dict(BASE, S0=10_000.0)
    discounted_strike = 100.0 * np.exp(-0.05)
    assert abs(bs_call_price(**params) - (10_000.0 - discounted_strike)) < 1e-6


def test_bs_deep_otm_call_is_worthless():
    assert bs_call_price(**dict(BASE, S0=1.0)) < 1e-10


def test_call_price_monotone_increasing_in_spot():
    prices = [bs_call_price(**dict(BASE, S0=s)) for s in (80, 90, 100, 110, 120)]
    assert all(b > a for a, b in zip(prices, prices[1:]))


def test_call_price_monotone_increasing_in_volatility():
    prices = [bs_call_price(**dict(BASE, sigma=s)) for s in (0.1, 0.2, 0.3, 0.4)]
    assert all(b > a for a, b in zip(prices, prices[1:]))


def test_price_respects_no_arbitrage_bounds():
    for s0 in (50.0, 100.0, 150.0):
        c = bs_call_price(**dict(BASE, S0=s0))
        lower = max(s0 - 100.0 * np.exp(-0.05), 0.0)
        assert lower - 1e-12 <= c <= s0 + 1e-12


def test_bs_delta_matches_finite_difference_of_price():
    h = 1e-4
    fd = (
        bs_call_price(**dict(BASE, S0=100.0 + h))
        - bs_call_price(**dict(BASE, S0=100.0 - h))
    ) / (2 * h)
    assert abs(fd - bs_call_delta(**BASE)) < 1e-7


def test_put_delta_equals_call_delta_minus_one():
    assert abs(bs_put_delta(**BASE) - (bs_call_delta(**BASE) - 1.0)) < 1e-12


def test_bs_vega_matches_finite_difference_of_price():
    h = 1e-5
    fd = (
        bs_call_price(**dict(BASE, sigma=0.2 + h))
        - bs_call_price(**dict(BASE, sigma=0.2 - h))
    ) / (2 * h)
    assert abs(fd - bs_vega(**BASE)) < 1e-5


def test_vega_is_identical_for_call_and_put():
    h = 1e-5
    fd_put = (
        bs_put_price(**dict(BASE, sigma=0.2 + h))
        - bs_put_price(**dict(BASE, sigma=0.2 - h))
    ) / (2 * h)
    assert abs(fd_put - bs_vega(**BASE)) < 1e-5


def test_bs_gamma_matches_second_difference_of_price():
    h = 1e-2
    second = (
        bs_call_price(**dict(BASE, S0=100.0 + h))
        - 2 * bs_call_price(**BASE)
        + bs_call_price(**dict(BASE, S0=100.0 - h))
    ) / h**2
    assert abs(second - bs_gamma(**BASE)) < 1e-6


def test_gamma_and_vega_are_positive():
    assert bs_gamma(**BASE) > 0
    assert bs_vega(**BASE) > 0


@pytest.mark.parametrize(
    "bad",
    [dict(sigma=-0.1), dict(T=-1.0), dict(S0=-100.0), dict(K=-100.0)],
)
def test_invalid_inputs_raise_value_error(bad):
    with pytest.raises(ValueError):
        bs_call_price(**dict(BASE, **bad))


def test_bs_theta_matches_finite_difference_in_time():
    from mcpricer.black_scholes import bs_theta

    h = 1e-5
    fd = -(
        bs_call_price(**dict(BASE, T=1.0 + h)) - bs_call_price(**dict(BASE, T=1.0 - h))
    ) / (2 * h)
    assert abs(fd - bs_theta(**BASE, option_type="call")) < 1e-5


def test_call_theta_is_negative_for_atm_option():
    from mcpricer.black_scholes import bs_theta

    assert bs_theta(**BASE, option_type="call") < 0


def test_bs_rho_matches_finite_difference_in_rate():
    from mcpricer.black_scholes import bs_rho

    h = 1e-6
    fd = (
        bs_call_price(**dict(BASE, r=0.05 + h)) - bs_call_price(**dict(BASE, r=0.05 - h))
    ) / (2 * h)
    assert abs(fd - bs_rho(**BASE, option_type="call")) < 1e-4


def test_bs_price_dispatch_matches_direct_calls():
    from mcpricer.black_scholes import bs_price

    assert bs_price(**BASE, option_type="call") == bs_call_price(**BASE)
    assert bs_price(**BASE, option_type="put") == bs_put_price(**BASE)


def test_unknown_option_type_raises():
    from mcpricer.black_scholes import bs_price

    with pytest.raises(ValueError):
        bs_price(**BASE, option_type="straddle")


def test_prices_broadcast_over_a_strike_ladder():
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    prices = bs_call_price(S0=100.0, K=strikes, r=0.05, sigma=0.2, T=1.0)
    assert prices.shape == strikes.shape
    assert abs(prices[2] - bs_call_price(**BASE)) < 1e-12
