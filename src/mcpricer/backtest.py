"""Delta-hedged straddle backtest on real S&P 500 data.

Every number the rest of this project produces is checked against a known
correct answer. This module is the opposite: a position is opened, hedged with
the engine's own deltas, and marked against realised market prices, so the
output is a P&L that can be wrong in ways no oracle will catch.

Design. Historical option chains are not freely available, so each cycle sells
a one-month at-the-money straddle struck at the index level and marked at the
VIX, the market's own 30-day implied volatility. The position is delta hedged
with the underlying at a configurable frequency and held to expiry. Limitations
of that construction are listed in the README; the spot path, the implied
volatility and the discount rate are all real observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from mcpricer.black_scholes import (
    bs_call_delta,
    bs_call_price,
    bs_put_delta,
    bs_put_price,
)

__all__ = ["BacktestResult", "straddle_value", "straddle_delta", "backtest_straddle"]

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    cycles: pd.DataFrame
    stats: dict[str, float]


def straddle_value(spot: float, strike: float, r: float, sigma: float, T: float) -> float:
    """Value of one call plus one put at the same strike."""
    return float(bs_call_price(spot, strike, r, sigma, T)) + float(
        bs_put_price(spot, strike, r, sigma, T)
    )


def straddle_delta(spot: float, strike: float, r: float, sigma: float, T: float) -> float:
    """Delta of the straddle, 2 N(d1) - 1. Near zero at the money, and it is
    the drift of that number that forces rehedging."""
    return float(bs_call_delta(spot, strike, r, sigma, T)) + float(
        bs_put_delta(spot, strike, r, sigma, T)
    )


def backtest_straddle(
    data: pd.DataFrame,
    *,
    holding_days: int = 21,
    rehedge_every: int = 1,
    direction: str = "short",
    hedge: bool = True,
    iv_offset: float = 0.0,
    hedge_cost_bps: float = 0.0,
    delta_fn: Callable[[float, float, float, float, float], float] | None = None,
) -> BacktestResult:
    """Roll a delta-hedged at-the-money straddle and record the P&L.

    P&L on each day is the mark-to-market change in the option position plus
    the return on the shares held into that day:

        pnl_t = sign (V_t - V_{t-1}) + h_{t-1} (S_t - S_{t-1})

    where h is set to cancel the option delta at each rehedge and held fixed in
    between. Everything is divided by the strike, so results are fractions of
    the notional and comparable across a decade of index levels.

    iv_offset is subtracted from the VIX everywhere. The VIX is a strip across
    strikes, so it sits above the at-the-money implied volatility that a
    straddle actually trades at; a negative offset tests whether the result
    survives selling at a realistically lower volatility.

    hedge_cost_bps charges each rehedge on the notional actually traded, which
    is what makes rehedging frequency a tradeoff rather than a free improvement.
    """
    if direction not in {"short", "long"}:
        raise ValueError("direction must be 'short' or 'long'")
    if holding_days < 2:
        raise ValueError("holding_days must be at least 2")
    if rehedge_every < 1:
        raise ValueError("rehedge_every must be at least 1")
    if len(data) < holding_days + 1:
        raise ValueError("not enough observations for a single cycle")

    sign = -1.0 if direction == "short" else 1.0
    delta_of = delta_fn if delta_fn is not None else straddle_delta

    spot = data["spot"].to_numpy(dtype=float)
    iv = np.maximum(data["iv"].to_numpy(dtype=float) + iv_offset, 1e-4)
    rate = data["rate"].to_numpy(dtype=float)
    dates = data.index

    daily_rows: list[dict[str, object]] = []
    cycle_rows: list[dict[str, object]] = []

    for start in range(0, len(data) - holding_days, holding_days):
        strike = spot[start]
        entry_iv = iv[start]
        shares = 0.0
        cycle_pnl = 0.0

        previous_value = straddle_value(
            spot[start], strike, rate[start], iv[start], holding_days / TRADING_DAYS
        )
        if hedge:
            shares = -sign * delta_of(
                spot[start], strike, rate[start], iv[start], holding_days / TRADING_DAYS
            )

        for step in range(1, holding_days + 1):
            index = start + step
            maturity = (holding_days - step) / TRADING_DAYS
            value = straddle_value(spot[index], strike, rate[index], iv[index], maturity)

            option_pnl = sign * (value - previous_value)
            hedge_pnl = shares * (spot[index] - spot[index - 1])
            pnl = (option_pnl + hedge_pnl) / strike

            daily_rows.append(
                {
                    "date": dates[index],
                    "pnl": pnl,
                    "option_pnl": option_pnl / strike,
                    "hedge_pnl": hedge_pnl / strike,
                    "spot": spot[index],
                    "iv": iv[index],
                    "shares": shares,
                    "cost": 0.0,
                }
            )

            previous_value = value
            cost = 0.0
            if hedge and maturity > 0 and step % rehedge_every == 0:
                target = -sign * delta_of(
                    spot[index], strike, rate[index], iv[index], maturity
                )
                cost = (
                    abs(target - shares)
                    * spot[index]
                    * hedge_cost_bps
                    * 1e-4
                    / strike
                )
                shares = target
            pnl -= cost
            cycle_pnl += pnl
            daily_rows[-1]["pnl"] = pnl
            daily_rows[-1]["cost"] = cost

        window = spot[start : start + holding_days + 1]
        realised = float(
            np.std(np.diff(np.log(window)), ddof=1) * np.sqrt(TRADING_DAYS)
        )
        cycle_rows.append(
            {
                "entry_date": dates[start],
                "expiry_date": dates[start + holding_days],
                "strike": strike,
                "entry_iv": entry_iv,
                "realised_vol": realised,
                "variance_risk_premium": entry_iv - realised,
                "premium": previous_value / strike if False else None,
                "pnl": cycle_pnl,
            }
        )

    daily = pd.DataFrame(daily_rows).set_index("date")
    cycles = pd.DataFrame(cycle_rows).drop(columns=["premium"])
    return BacktestResult(daily, cycles, _summarise(daily, cycles))


def _summarise(daily: pd.DataFrame, cycles: pd.DataFrame) -> dict[str, float]:
    pnl = daily["pnl"]
    cumulative = pnl.cumsum()
    drawdown = (cumulative - cumulative.cummax()).min()
    volatility = pnl.std(ddof=1)
    return {
        "n_cycles": float(len(cycles)),
        "total_return": float(pnl.sum()),
        "annualised_return": float(pnl.mean() * TRADING_DAYS),
        "annualised_volatility": float(volatility * np.sqrt(TRADING_DAYS)),
        "sharpe": float(pnl.mean() / volatility * np.sqrt(TRADING_DAYS))
        if volatility > 0
        else float("nan"),
        "max_drawdown": float(drawdown),
        "hit_rate": float((cycles["pnl"] > 0).mean()),
        "worst_cycle": float(cycles["pnl"].min()),
        "best_cycle": float(cycles["pnl"].max()),
        "worst_day": float(pnl.min()),
        "mean_entry_iv": float(cycles["entry_iv"].mean()),
        "mean_realised_vol": float(cycles["realised_vol"].mean()),
        "mean_variance_risk_premium": float(cycles["variance_risk_premium"].mean()),
    }
