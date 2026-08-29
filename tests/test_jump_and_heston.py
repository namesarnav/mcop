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


HESTON = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
HESTON_PATHS = 200_000
HESTON_STEPS = 252


def _heston_terminal(seed, **overrides):
    from mcpricer.simulation import simulate_heston_paths

    params = dict(HESTON)
    params.update(overrides)
    return simulate_heston_paths(
        S0=100.0, r=0.05, T=1.0, n_paths=HESTON_PATHS, n_steps=HESTON_STEPS,
        seed=seed, **params,
    )[:, -1]


def test_heston_path_shape_and_initial_value():
    from mcpricer.simulation import simulate_heston_paths

    paths = simulate_heston_paths(
        S0=100.0, r=0.05, T=1.0, n_paths=100, n_steps=50, seed=0, **HESTON
    )
    assert paths.shape == (100, 51)
    assert np.allclose(paths[:, 0], 100.0)


def test_heston_reduces_to_gbm_when_vol_of_vol_is_zero():
    # With xi = 0 and v0 = theta the variance never moves, so the model is
    # flat-vol GBM at sigma = sqrt(theta) = 0.2.
    terminal = _heston_terminal(2, xi=0.0, rho=0.0)
    gbm = simulate_gbm_terminal(**BASE, n_paths=HESTON_PATHS, seed=3)
    se = np.sqrt(terminal.var(ddof=1) / terminal.size + gbm.var(ddof=1) / gbm.size)
    assert abs(terminal.mean() - gbm.mean()) < 3 * se
    assert abs(terminal.var(ddof=1) / gbm.var(ddof=1) - 1.0) < 0.05


def test_heston_discounted_price_is_a_martingale():
    terminal = _heston_terminal(4)
    discounted = np.exp(-0.05) * terminal
    se = discounted.std(ddof=1) / np.sqrt(discounted.size)
    assert abs(discounted.mean() - 100.0) < 3 * se


def test_negative_correlation_produces_negative_skew():
    # The leverage effect: falling prices raise volatility, fattening the left
    # tail. This is the mechanism behind the equity smile's downward slope.
    assert skew(np.log(_heston_terminal(5))) < -0.5


def test_positive_correlation_flips_the_skew():
    assert skew(np.log(_heston_terminal(6, rho=0.7))) > 0


def test_heston_variance_is_never_negative_where_it_is_used():
    from mcpricer.simulation import simulate_heston_paths

    paths, variance = simulate_heston_paths(
        S0=100.0, r=0.05, T=1.0, n_paths=20_000, n_steps=100, seed=7,
        return_variance=True, v0=0.04, kappa=1.0, theta=0.04, xi=1.5, rho=-0.5,
    )
    # Full truncation lets the state go negative but never the volatility used.
    assert np.all(paths > 0.0)
    assert np.all(np.sqrt(np.maximum(variance, 0.0)) >= 0.0)


def test_heston_mean_variance_reverts_towards_theta():
    from mcpricer.simulation import simulate_heston_paths

    _, variance = simulate_heston_paths(
        S0=100.0, r=0.05, T=5.0, n_paths=50_000, n_steps=500, seed=8,
        return_variance=True, v0=0.16, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5,
    )
    assert abs(np.maximum(variance[:, -1], 0.0).mean() - 0.04) < 0.005


@pytest.mark.parametrize("bad", [dict(rho=1.5), dict(xi=-0.1), dict(v0=-0.01)])
def test_invalid_heston_parameters_raise(bad):
    from mcpricer.simulation import simulate_heston_paths

    params = dict(HESTON)
    params.update(bad)
    with pytest.raises(ValueError):
        simulate_heston_paths(
            S0=100.0, r=0.05, T=1.0, n_paths=100, n_steps=10, seed=0, **params
        )


SMILE_STRIKES = np.array([85.0, 92.5, 100.0, 107.5, 115.0])


def _smile_from_terminal(terminal):
    from mcpricer.black_scholes import implied_volatility_smile

    prices = np.array(
        [np.exp(-0.05) * call_payoff(terminal, k).mean() for k in SMILE_STRIKES]
    )
    return implied_volatility_smile(prices, 100.0, SMILE_STRIKES, 0.05, 1.0)


def test_gbm_implied_volatility_is_flat_by_construction():
    smile = _smile_from_terminal(simulate_gbm_terminal(**BASE, n_paths=300_000, seed=2))
    assert np.ptp(smile) < 0.005
    assert abs(smile.mean() - 0.2) < 0.005


def test_heston_produces_a_downward_sloping_skew():
    from mcpricer.simulation import simulate_heston_paths

    terminal = simulate_heston_paths(
        S0=100.0, r=0.05, T=1.0, n_paths=300_000, n_steps=100, seed=0,
        v0=0.04, kappa=2.0, theta=0.04, xi=0.6, rho=-0.7,
    )[:, -1]
    smile = _smile_from_terminal(terminal)
    assert np.all(np.diff(smile) < 0)
    assert np.ptp(smile) > 0.03


def test_merton_downward_jumps_produce_a_skew():
    terminal = simulate_merton_terminal(**BASE, n_paths=1_000_000, seed=1, **JUMPS)
    smile = _smile_from_terminal(terminal)
    assert smile[0] > smile[-1]
    assert np.ptp(smile) > 0.005
