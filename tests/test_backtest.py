import numpy as np
import pandas as pd
import pytest

from mcpricer.backtest import backtest_straddle, straddle_delta, straddle_value
from mcpricer.data import load_market_data


def flat_market(n_days=22, spot=100.0, iv=0.2, rate=0.0):
    index = pd.bdate_range("2024-01-01", periods=n_days, name="date")
    return pd.DataFrame(
        {"spot": np.full(n_days, spot), "iv": np.full(n_days, iv), "rate": np.full(n_days, rate)},
        index=index,
    )


def test_straddle_value_is_call_plus_put():
    from mcpricer.black_scholes import bs_call_price, bs_put_price

    expected = bs_call_price(100.0, 100.0, 0.05, 0.2, 1.0) + bs_put_price(
        100.0, 100.0, 0.05, 0.2, 1.0
    )
    assert abs(straddle_value(100.0, 100.0, 0.05, 0.2, 1.0) - expected) < 1e-12


def test_atm_straddle_delta_is_small_and_positive():
    delta = straddle_delta(100.0, 100.0, 0.05, 0.2, 1.0)
    assert 0.0 < delta < 0.4


def test_short_straddle_on_a_motionless_market_collects_the_full_premium():
    # Spot never moves, so the hedge earns nothing and realised volatility is
    # zero: the short position keeps exactly the premium it sold.
    data = flat_market()
    result = backtest_straddle(data, holding_days=21)
    premium = straddle_value(100.0, 100.0, 0.0, 0.2, 21 / 252) / 100.0
    assert abs(result.stats["total_return"] - premium) < 1e-12
    assert abs(result.daily["hedge_pnl"]).max() < 1e-12


def test_long_and_short_are_exact_mirrors():
    data = load_market_data().head(200)
    short = backtest_straddle(data, direction="short")
    long = backtest_straddle(data, direction="long")
    assert np.allclose(short.daily["pnl"], -long.daily["pnl"])


def test_hedging_reduces_volatility_on_real_data():
    data = load_market_data()
    hedged = backtest_straddle(data)
    unhedged = backtest_straddle(data, hedge=False)
    assert hedged.stats["annualised_volatility"] < unhedged.stats["annualised_volatility"]
    assert abs(hedged.stats["max_drawdown"]) < abs(unhedged.stats["max_drawdown"])


def test_hedging_only_adds_a_hedge_leg():
    # The option mark-to-market is identical either way; hedging changes the
    # result solely through the shares held.
    data = load_market_data().head(300)
    hedged = backtest_straddle(data)
    unhedged = backtest_straddle(data, hedge=False)
    assert np.allclose(hedged.daily["option_pnl"], unhedged.daily["option_pnl"])
    assert np.allclose(unhedged.daily["hedge_pnl"], 0.0)


def test_less_frequent_rehedging_raises_volatility():
    data = load_market_data()
    daily = backtest_straddle(data, rehedge_every=1)
    weekly = backtest_straddle(data, rehedge_every=5)
    assert weekly.stats["annualised_volatility"] > daily.stats["annualised_volatility"]


def test_hedge_costs_are_zero_by_default_and_reduce_returns_when_charged():
    data = load_market_data().head(300)
    free = backtest_straddle(data)
    charged = backtest_straddle(data, hedge_cost_bps=5.0)
    assert (free.daily["cost"] == 0.0).all()
    assert (charged.daily["cost"] >= 0.0).all()
    assert charged.stats["total_return"] < free.stats["total_return"]


def test_a_lower_selling_volatility_reduces_the_edge():
    data = load_market_data()
    rich = backtest_straddle(data)
    cheap = backtest_straddle(data, iv_offset=-0.03)
    assert cheap.stats["annualised_return"] < rich.stats["annualised_return"]


def test_realised_volatility_spiked_in_the_covid_cycle():
    # Sanity check on the cycle bookkeeping: the worst cycle should be the one
    # containing the February 2020 crash.
    data = load_market_data()
    cycles = backtest_straddle(data).cycles
    worst = cycles.loc[cycles["pnl"].idxmin()]
    assert worst["entry_date"].year == 2020
    assert worst["realised_vol"] > 0.5


def test_variance_risk_premium_is_positive_on_average():
    cycles = backtest_straddle(load_market_data()).cycles
    assert cycles["variance_risk_premium"].mean() > 0


def test_monte_carlo_deltas_hedge_the_same_as_closed_form_deltas():
    from mcpricer.greeks import mc_delta_pathwise

    def mc_straddle_delta(spot, strike, r, sigma, T):
        if T <= 0:
            return 0.0
        call = mc_delta_pathwise(spot, strike, r, sigma, T, 200_000, "call", seed=0)
        put = mc_delta_pathwise(spot, strike, r, sigma, T, 200_000, "put", seed=1)
        return call.value + put.value

    data = load_market_data().head(22)
    closed = backtest_straddle(data)
    monte_carlo = backtest_straddle(data, delta_fn=mc_straddle_delta)
    assert abs(closed.stats["total_return"] - monte_carlo.stats["total_return"]) < 5e-4


@pytest.mark.parametrize(
    "bad", [dict(direction="flat"), dict(holding_days=1), dict(rehedge_every=0)]
)
def test_invalid_backtest_inputs_raise(bad):
    with pytest.raises(ValueError):
        backtest_straddle(load_market_data().head(100), **bad)


def test_too_little_data_raises():
    with pytest.raises(ValueError):
        backtest_straddle(flat_market(n_days=5), holding_days=21)
