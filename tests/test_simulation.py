import time

import numpy as np
import pytest

from mcpricer.simulation import simulate_gbm_paths, simulate_gbm_terminal

S0, R, SIGMA, T = 100.0, 0.05, 0.2, 1.0


def test_terminal_price_zero_volatility_is_deterministic():
    terminal = simulate_gbm_terminal(S0=S0, r=R, sigma=0.0, T=T, n_paths=1000, seed=0)
    assert np.allclose(terminal, S0 * np.exp(R * T), atol=1e-12)


def test_terminal_price_zero_maturity_returns_spot():
    terminal = simulate_gbm_terminal(S0=S0, r=R, sigma=SIGMA, T=0.0, n_paths=100, seed=0)
    assert np.allclose(terminal, S0, atol=1e-12)


def test_terminal_price_mean_matches_theory():
    # E[S_T] = S0 e^{rT}
    terminal = simulate_gbm_terminal(
        S0=S0, r=R, sigma=SIGMA, T=T, n_paths=1_000_000, seed=42
    )
    expected = S0 * np.exp(R * T)
    se = terminal.std(ddof=1) / np.sqrt(terminal.size)
    assert abs(terminal.mean() - expected) < 3 * se


def test_discounted_price_is_a_martingale():
    # E[e^{-rT} S_T] = S0
    terminal = simulate_gbm_terminal(
        S0=S0, r=R, sigma=SIGMA, T=T, n_paths=1_000_000, seed=7
    )
    discounted = np.exp(-R * T) * terminal
    se = discounted.std(ddof=1) / np.sqrt(discounted.size)
    assert abs(discounted.mean() - S0) < 3 * se


def test_terminal_price_variance_matches_theory():
    # Var(S_T) = S0^2 e^{2rT} (e^{sigma^2 T} - 1)
    terminal = simulate_gbm_terminal(
        S0=S0, r=R, sigma=SIGMA, T=T, n_paths=2_000_000, seed=11
    )
    expected = S0**2 * np.exp(2 * R * T) * (np.exp(SIGMA**2 * T) - 1)
    assert abs(terminal.var(ddof=1) / expected - 1.0) < 0.01


def test_log_returns_have_the_ito_corrected_drift():
    # log(S_T/S0) ~ N((r - sigma^2/2)T, sigma^2 T)
    terminal = simulate_gbm_terminal(
        S0=S0, r=R, sigma=SIGMA, T=T, n_paths=1_000_000, seed=3
    )
    log_returns = np.log(terminal / S0)
    expected_mean = (R - 0.5 * SIGMA**2) * T
    se = log_returns.std(ddof=1) / np.sqrt(log_returns.size)
    assert abs(log_returns.mean() - expected_mean) < 3 * se
    assert abs(log_returns.std(ddof=1) - SIGMA * np.sqrt(T)) < 1e-3


def test_terminal_prices_are_strictly_positive():
    terminal = simulate_gbm_terminal(
        S0=S0, r=R, sigma=2.0, T=5.0, n_paths=200_000, seed=5
    )
    assert np.all(terminal > 0.0)


def test_simulation_is_reproducible_under_a_fixed_seed():
    a = simulate_gbm_terminal(S0=S0, r=R, sigma=SIGMA, T=T, n_paths=1000, seed=123)
    b = simulate_gbm_terminal(S0=S0, r=R, sigma=SIGMA, T=T, n_paths=1000, seed=123)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_draws():
    a = simulate_gbm_terminal(S0=S0, r=R, sigma=SIGMA, T=T, n_paths=1000, seed=1)
    b = simulate_gbm_terminal(S0=S0, r=R, sigma=SIGMA, T=T, n_paths=1000, seed=2)
    assert not np.array_equal(a, b)


def test_terminal_simulation_is_vectorised():
    # Unreachable with a Python-level loop over paths.
    start = time.perf_counter()
    simulate_gbm_terminal(S0=S0, r=R, sigma=SIGMA, T=T, n_paths=5_000_000, seed=0)
    assert time.perf_counter() - start < 2.0


def test_full_path_shape_correct():
    paths = simulate_gbm_paths(
        S0=S0, r=R, sigma=SIGMA, T=T, n_paths=100, n_steps=50, seed=0
    )
    assert paths.shape == (100, 51)
    assert np.allclose(paths[:, 0], S0)


def test_full_paths_zero_volatility_follow_the_growth_curve():
    paths = simulate_gbm_paths(S0=S0, r=R, sigma=0.0, T=T, n_paths=10, n_steps=4, seed=0)
    grid = np.linspace(0.0, T, 5)
    assert np.allclose(paths, S0 * np.exp(R * grid), atol=1e-12)


def test_full_path_terminal_matches_the_terminal_sampler():
    # Log-space stepping is exact, so time-stepping adds no discretisation error.
    paths = simulate_gbm_paths(
        S0=S0, r=R, sigma=SIGMA, T=T, n_paths=500_000, n_steps=64, seed=9
    )
    stepped_terminal = paths[:, -1]
    expected = S0 * np.exp(R * T)
    se = stepped_terminal.std(ddof=1) / np.sqrt(stepped_terminal.size)
    assert abs(stepped_terminal.mean() - expected) < 3 * se


def test_full_paths_are_strictly_positive():
    paths = simulate_gbm_paths(
        S0=S0, r=R, sigma=1.5, T=2.0, n_paths=10_000, n_steps=200, seed=4
    )
    assert np.all(paths > 0.0)


@pytest.mark.parametrize(
    "bad",
    [dict(sigma=-0.1), dict(T=-1.0), dict(S0=-100.0), dict(n_paths=0), dict(n_paths=-5)],
)
def test_invalid_terminal_inputs_raise(bad):
    kwargs = dict(S0=S0, r=R, sigma=SIGMA, T=T, n_paths=1000, seed=0)
    with pytest.raises(ValueError):
        simulate_gbm_terminal(**dict(kwargs, **bad))


def test_non_positive_step_count_raises():
    with pytest.raises(ValueError):
        simulate_gbm_paths(S0=S0, r=R, sigma=SIGMA, T=T, n_paths=10, n_steps=0, seed=0)
