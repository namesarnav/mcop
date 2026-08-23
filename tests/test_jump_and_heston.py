import numpy as np
import pytest
from scipy.stats import skew

from mcpricer.black_scholes import bs_call_price, merton_call_price
from mcpricer.payoffs import call_payoff
from mcpricer.simulation import simulate_gbm_terminal, simulate_merton_terminal

BASE = dict(S0=100.0, r=0.05, sigma=0.2, T=1.0)
JUMPS = dict(jump_intensity=0.75, jump_mean=-0.05, jump_std=0.15)
N = 2_000_000


def test_merton_closed_form_reduces_to_black_scholes_without_jumps():
    price = merton_call_price(
        S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0,
        jump_intensity=0.0, jump_mean=0.0, jump_std=0.0,
    )
    assert abs(price - bs_call_price(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)) < 1e-12


def test_merton_simulation_reduces_to_gbm_without_jumps():
    merton = simulate_merton_terminal(
        **BASE, n_paths=N, seed=0, jump_intensity=0.0, jump_mean=0.0, jump_std=0.0
    )
    gbm = simulate_gbm_terminal(**BASE, n_paths=N, seed=1)
    se = np.sqrt(merton.var(ddof=1) / N + gbm.var(ddof=1) / N)
    assert abs(merton.mean() - gbm.mean()) < 3 * se
    assert abs(merton.var(ddof=1) / gbm.var(ddof=1) - 1.0) < 0.02


def test_merton_discounted_price_is_a_martingale():
    # The -lambda k compensator in the drift is what makes this hold.
    terminal = simulate_merton_terminal(**BASE, n_paths=N, seed=2, **JUMPS)
    discounted = np.exp(-BASE["r"] * BASE["T"]) * terminal
    se = discounted.std(ddof=1) / np.sqrt(N)
    assert abs(discounted.mean() - BASE["S0"]) < 3 * se


def test_merton_mc_price_matches_the_closed_form():
    terminal = simulate_merton_terminal(**BASE, n_paths=N, seed=3, **JUMPS)
    discounted = np.exp(-BASE["r"] * BASE["T"]) * call_payoff(terminal, 100.0)
    price = discounted.mean()
    se = discounted.std(ddof=1) / np.sqrt(N)
    closed = merton_call_price(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, **JUMPS)
    assert abs(price - closed) < 3 * se


def test_jumps_add_variance_at_matched_diffusive_volatility():
    merton = simulate_merton_terminal(**BASE, n_paths=N, seed=4, **JUMPS)
    gbm = simulate_gbm_terminal(**BASE, n_paths=N, seed=5)
    assert merton.var(ddof=1) > gbm.var(ddof=1)


def test_negative_mean_jumps_produce_a_fatter_left_tail():
    merton = simulate_merton_terminal(**BASE, n_paths=N, seed=6, **JUMPS)
    gbm = simulate_gbm_terminal(**BASE, n_paths=N, seed=7)
    assert skew(np.log(merton)) < skew(np.log(gbm))


def test_jumps_raise_the_option_price_above_black_scholes():
    closed = merton_call_price(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, **JUMPS)
    assert closed > bs_call_price(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)


@pytest.mark.parametrize("bad", [dict(jump_intensity=-1.0), dict(jump_std=-0.1)])
def test_invalid_jump_parameters_raise(bad):
    params = dict(BASE, n_paths=1000, seed=0)
    params.update(JUMPS)
    params.update(bad)
    with pytest.raises(ValueError):
        simulate_merton_terminal(**params)
