"""Input guards: bad arguments must raise rather than return silent NaNs."""

import numpy as np
import pytest

from mcpricer.binomial import binomial_price
from mcpricer.black_scholes import bs_theta, implied_volatility
from mcpricer.greeks import (
    mc_delta_finite_difference,
    mc_delta_pathwise,
    mc_gamma_finite_difference,
    mc_vega_pathwise,
)
from mcpricer.payoffs import get_payoff
from mcpricer.pricing import PriceEstimate, lsm_american_price
from mcpricer.simulation import simulate_gbm_terminal
from mcpricer.variance_reduction import (
    mc_european_price_antithetic,
    mc_european_price_control_variate,
    variance_reduction_pct,
)

BASE = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)


def test_seed_and_rng_are_mutually_exclusive():
    with pytest.raises(ValueError):
        simulate_gbm_terminal(
            S0=100.0, r=0.05, sigma=0.2, T=1.0, n_paths=10,
            seed=1, rng=np.random.default_rng(2),
        )


def test_unknown_payoff_type_raises():
    with pytest.raises(ValueError):
        get_payoff("barrier")


def test_confidence_level_must_be_a_probability():
    with pytest.raises(ValueError):
        PriceEstimate(10.0, 0.1).confidence_interval(1.5)


def test_theta_accepts_puts():
    assert bs_theta(**BASE, option_type="put") < 0


@pytest.mark.parametrize("bad", [dict(n_paths=0), dict(n_paths=3)])
def test_antithetic_input_guards(bad):
    params = dict(BASE, n_paths=10, seed=0)
    params.update(bad)
    with pytest.raises(ValueError):
        mc_european_price_antithetic(**params)


def test_antithetic_rejects_negative_volatility():
    with pytest.raises(ValueError):
        mc_european_price_antithetic(**dict(BASE, sigma=-0.2), n_paths=10, seed=0)


@pytest.mark.parametrize("pilot", [0.0, 1.0, 1.5])
def test_control_variate_pilot_fraction_must_be_a_proper_fraction(pilot):
    with pytest.raises(ValueError):
        mc_european_price_control_variate(
            **BASE, n_paths=1000, pilot_fraction=pilot, seed=0
        )


def test_control_variate_rejects_empty_sample():
    with pytest.raises(ValueError):
        mc_european_price_control_variate(**BASE, n_paths=0, seed=0)


def test_control_variate_coefficient_is_zero_without_control_variance():
    # sigma = 0 makes S_T deterministic, so the control carries no information.
    estimate = mc_european_price_control_variate(
        **dict(BASE, sigma=0.0), n_paths=1000, seed=0
    )
    assert estimate.coefficient == 0.0


def test_variance_reduction_pct_requires_a_positive_baseline():
    with pytest.raises(ValueError):
        variance_reduction_pct(0.0, 0.01)


def test_greeks_reject_non_positive_path_counts():
    with pytest.raises(ValueError):
        mc_delta_finite_difference(**BASE, n_paths=0, seed=0)
    with pytest.raises(ValueError):
        mc_delta_pathwise(**BASE, n_paths=0, seed=0)


def test_gamma_rejects_a_non_positive_bump():
    with pytest.raises(ValueError):
        mc_gamma_finite_difference(**BASE, n_paths=1000, bump=0.0, seed=0)


def test_pathwise_vega_rejects_unknown_option_type():
    with pytest.raises(ValueError):
        mc_vega_pathwise(**BASE, n_paths=1000, option_type="digital", seed=0)


def test_binomial_rejects_unknown_option_type():
    with pytest.raises(ValueError):
        binomial_price(**BASE, n_steps=10, option_type="digital")


def test_binomial_rejects_a_step_size_that_breaks_the_probability():
    # A huge rate relative to the step volatility pushes p outside [0, 1].
    with pytest.raises(ValueError):
        binomial_price(S0=100.0, K=100.0, r=5.0, sigma=0.05, T=1.0, n_steps=2)


def test_lsm_rejects_an_invalid_polynomial_degree():
    with pytest.raises(ValueError):
        lsm_american_price(**BASE, n_paths=1000, n_steps=10, degree=0)


def test_implied_volatility_rejects_an_unknown_option_type():
    with pytest.raises(ValueError):
        implied_volatility(10.0, 100.0, 100.0, 0.05, 1.0, "digital")
