# Monte Carlo Options Pricer

A Monte Carlo engine for pricing European and American options — with variance
reduction, Greeks, and alternative asset dynamics — and a backtest that puts its
deltas to work hedging a real position on ten years of S&P 500 data.

The project has two halves that are deliberately different in kind. The **engine**
is validated against independent deterministic references: closed-form
Black-Scholes, Merton's series solution, a binomial tree. Every stochastic result
must land within three standard errors of a value that was computed without
randomness. The **backtest** has no oracle at all, which is the point of including
it — it produces a P&L that can be wrong in ways no reference value will catch.

202 tests from 164 test functions, 98% coverage on non-plotting modules. Every
figure and table below is reproducible via `python scripts/generate_report.py`.
Monte Carlo results come from fixed seeds; market results from CSVs cached under
`data/`.

---

## Contents

1. [Headline results](#1-headline-results)
2. [Architecture](#2-architecture)
3. [Module reference](#3-module-reference)
4. [Mathematical foundations](#4-mathematical-foundations)
5. [Simulation](#5-simulation)
6. [European pricing and convergence](#6-european-pricing-and-convergence)
7. [Variance reduction](#7-variance-reduction)
8. [Greeks](#8-greeks)
9. [American options](#9-american-options)
10. [Alternative dynamics and volatility smiles](#10-alternative-dynamics-and-volatility-smiles)
11. [The hedging backtest](#11-the-hedging-backtest)
12. [Validation methodology](#12-validation-methodology)
13. [Running it](#13-running-it)
14. [Limitations and next steps](#14-limitations-and-next-steps)

---

## 1. Headline results

Reference contract for all pricing results: `S0 = K = 100`, `r = 5%`,
`sigma = 20%`, `T = 1`. Black-Scholes call **10.4506**, put **5.5735**.

| Result | Value |
|---|---|
| MC call price, 1M paths | 10.4537 ± 0.0147 (0.2 se from closed form) |
| Variance removed by control variate | 83.9%, a 5.97× efficiency gain |
| Greeks vs. closed form (5 estimators) | all within 1.8 standard errors |
| Finite-difference error without common random numbers | ~1,800× worse |
| American put, Longstaff-Schwartz vs. binomial tree | 6.0846 ± 0.0101 vs. 6.0902 |
| Heston implied vol, strikes 80 → 120 | 24.5% → 14.5% (GBM flat at 19.9%) |
| Delta-hedged straddle Sharpe, hedged vs. unhedged | 1.31 vs. 0.36 |
| Max drawdown, hedged vs. unhedged | −6.3% vs. −32.0% |

---

## 2. Architecture

### Dependency layers

```
LAYER 0  — no internal dependencies
  black_scholes.py    ORACLE: closed-form prices, Greeks, Merton series, implied vol
  binomial.py         ORACLE: Cox-Ross-Rubinstein tree
  simulation.py       GBM, Merton jump-diffusion, Heston
  payoffs.py          vanilla payoff functions
  data.py             cached FRED market data

LAYER 1
  greeks.py           -> payoffs, simulation
  pricing.py          -> payoffs, simulation, black_scholes
  backtest.py         -> black_scholes

LAYER 2
  variance_reduction.py -> pricing, payoffs, simulation

LAYER 3
  reporting.py        -> everything
```

**The oracles sit at layer 0 and depend on nothing.** They cannot be contaminated
by the code they validate. This is what makes "the Monte Carlo engine is correct"
a checkable claim rather than an assertion: `black_scholes.py` and `binomial.py`
contain no randomness and no simulation, are tested first against exact textbook
values and model-free identities, and everything stochastic is then tested
against them.

### Design decisions

| Decision | Rationale |
|---|---|
| Simulate in log space | The GBM SDE has an exact solution in `log S`, so time-stepping introduces no discretisation error and prices cannot go negative |
| Prices always returned with a standard error | `PriceEstimate` is a `NamedTuple`; a Monte Carlo price without an error bar is not a result |
| Generator injection (`seed=` / `rng=`) | Reproducibility without global state; `SeedSequence.spawn` gives independent streams so estimator comparisons aren't an artefact of one shared draw |
| Payoffs as first-class callables | `get_payoff(option_type)` returns a function, so pricing code never branches on call/put |
| Validation at the boundary | Bad input raises immediately rather than producing a NaN that surfaces a million paths later |
| Full vectorisation | No Python-level loop over paths anywhere; a test asserts 5M paths in under 2 seconds |
| Plotting isolated in `reporting.py` | Excluded from coverage, so the 98% figure describes the maths |

### Patterns used

**Test oracle** (layer-0 analytic references), **dependency injection**
(`rng`, `delta_fn`), **strategy lookup** (`get_payoff`), **value objects**
(`PriceEstimate`, `GreekEstimate`, `ControlVariateEstimate`), **guard clauses**,
**seed trees**, **metamorphic testing** (`lambda=0` recovers GBM; `xi=0, v0=theta`
recovers GBM), and **property testing** (put-call parity, martingale property,
monotonicity, no-arbitrage bounds).

---

## 3. Module reference

### `black_scholes.py` — analytic oracle
```python
bs_call_price(S0, K, r, sigma, T)          # C = S0 N(d1) - K e^{-rT} N(d2)
bs_put_price(S0, K, r, sigma, T)           # P = K e^{-rT} N(-d2) - S0 N(-d1)
bs_call_delta / bs_put_delta               # N(d1) / N(d1) - 1
bs_vega / bs_gamma / bs_theta / bs_rho
bs_price(..., option_type)                 # dispatch
merton_call_price(..., jump_intensity, jump_mean, jump_std)
implied_volatility(price, S0, K, r, T, option_type)
implied_volatility_smile(prices, S0, strikes, r, T)
```
All functions broadcast over arrays, so a whole strike ladder prices in one call.
The degenerate case `sigma * sqrt(T) == 0` is handled explicitly — both `sigma=0`
and `T=0` make the payoff a deterministic function of the forward, so the price
is discounted intrinsic value and delta is a step function.

### `simulation.py` — asset dynamics
```python
gbm_terminal_from_normals(S0, r, sigma, T, z)   # z -> S_T, for CRN and antithetic
simulate_gbm_terminal(S0, r, sigma, T, n_paths, seed=, rng=)
simulate_gbm_paths(..., n_steps)                # (n_paths, n_steps + 1)
simulate_merton_terminal(..., jump_intensity, jump_mean, jump_std)
simulate_heston_paths(..., v0, kappa, theta, xi, rho, return_variance=False)
```

### `payoffs.py`
```python
call_payoff(S, K)      # max(S - K, 0)
put_payoff(S, K)       # max(K - S, 0)
get_payoff(option_type)
```

### `pricing.py`
```python
class PriceEstimate(NamedTuple):     # unpacks as (price, standard_error)
    price: float
    standard_error: float
    def confidence_interval(level=0.95) -> tuple[float, float]

mc_european_price(S0, K, r, sigma, T, n_paths, option_type, seed=, rng=)
mc_european_call_price / mc_european_put_price
convergence_study(..., path_counts)
lsm_american_price(..., n_steps, degree=3)
early_exercise_premium(...)
```

### `variance_reduction.py`
```python
mc_european_price_antithetic(...)                    # -> PriceEstimate
mc_european_price_control_variate(..., pilot_fraction=0.1)  # -> ControlVariateEstimate
variance_reduction_pct(se_naive, se_reduced)
efficiency_gain(se_naive, t_naive, se_reduced, t_reduced)
compare_estimators(...)                              # timed comparison table
```

### `greeks.py`
```python
class GreekEstimate(NamedTuple):
    value: float
    standard_error: float

mc_delta_finite_difference(..., bump=0.01, common_random_numbers=True)
mc_delta_pathwise(...)
mc_vega_finite_difference(..., bump=0.01)
mc_vega_pathwise(...)
mc_gamma_finite_difference(..., bump=1.0)
```

### `binomial.py`
```python
binomial_price(S0, K, r, sigma, T, n_steps, option_type, exercise)
binomial_european_call_price / binomial_european_put_price
binomial_american_call_price / binomial_american_put_price
```

### `data.py` and `backtest.py`
```python
load_market_data(data_dir=None) -> DataFrame[spot, iv, rate]
download_series(series_id, destination)              # refresh the cache

straddle_value(spot, strike, r, sigma, T)
straddle_delta(spot, strike, r, sigma, T)            # 2 N(d1) - 1
backtest_straddle(data, holding_days=21, rehedge_every=1, direction="short",
                  hedge=True, iv_offset=0.0, hedge_cost_bps=0.0, delta_fn=None)
```

---

## 4. Mathematical foundations

### Risk-neutral pricing

By the fundamental theorem of asset pricing, absence of arbitrage implies a
measure `Q` under which every discounted tradable price is a martingale. The
option price is then the discounted expectation of its payoff:

```
C = e^{-rT} E_Q[payoff(S_T)]
```

The asset drifts at `r`, not at its real-world expected return, because
`e^{-rt} S_t` must be driftless under `Q`. Preferences drop out — a delta-hedged
option portfolio is instantaneously riskless, so it must earn `r`. This is why
Monte Carlo works: pricing becomes computing an expectation, and expectations are
exactly what sampling estimates.

### Geometric Brownian motion

Starting from `dS = r S dt + sigma S dW` and applying Itô's lemma to `log S`:

```
d(log S) = (r - sigma^2/2) dt + sigma dW
```

The `-sigma^2/2` is the Itô correction, arising because `(dW)^2 = dt`.
Integrating gives an exact solution:

```
S_T = S0 exp[(r - sigma^2/2) T + sigma sqrt(T) Z],    Z ~ N(0, 1)
E[S_T]   = S0 e^{rT}
Var(S_T) = S0^2 e^{2rT} (e^{sigma^2 T} - 1)
```

### Black-Scholes

```
C  = S0 N(d1) - K e^{-rT} N(d2)
d1 = [ln(S0/K) + (r + sigma^2/2) T] / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
```

`N(d2)` is the probability of finishing in the money under `Q`. `N(d1)` is delta,
and also the exercise probability under the stock-numeraire measure — so
`S0 N(d1)` is the expected value of the stock received conditional on exercise.

**Put-call parity** `C - P = S0 - K e^{-rT}` follows from two portfolios with
identical terminal payoff `S_T - K`. It is model-free, which is why the test
suite asserts it to `1e-10` rather than statistically. Differentiating it gives
`vega_call = vega_put` and `delta_put = delta_call - 1`, both of which are tested.

### The Monte Carlo estimator

```
C_hat = e^{-rT} (1/N) sum_i payoff(S_T^i)
SE    = e^{-rT} sigma_payoff / sqrt(N)
```

By the CLT the error is `O(N^{-1/2})`, independent of dimension — the reason
Monte Carlo beats trees and PDE methods in high dimensions. Halving the error
costs 4× the paths; one more decimal digit costs 100×. **That cost curve is the
entire motivation for variance reduction:** you cannot brute-force precision, so
you attack `sigma_payoff` instead of `N`.

---

## 5. Simulation

### Log-space GBM

Because the solution above is exact, European payoffs need no intermediate time
steps at all — `S_T` is sampled in a single vectorised draw. The full
time-stepped path is generated only for early exercise and stochastic volatility.
Simulating `log S` and exponentiating makes a non-positive price structurally
impossible; a naive multiplicative Euler scheme can go negative.

Tested by: `sigma=0` collapsing every path onto `S0 e^{rT}` exactly, `T=0`
returning spot, `E[S_T]` and `Var(S_T)` matching theory, the discounted price
being a martingale, the Itô drift tested **separately** from `E[S_T]` (a sign
error in `-sigma^2/2` fails the two differently), strict positivity at
`sigma = 2.0, T = 5`, and 5M paths in under 2 seconds as a vectorisation proxy.

### Merton jump-diffusion

```
S_T = S0 exp[(r - lambda k - sigma^2/2) T + sigma sqrt(T) Z + sum_{i=1}^{N_T} Y_i]
N_T ~ Poisson(lambda T),  Y_i ~ N(mu_J, sigma_J^2),  k = e^{mu_J + sigma_J^2/2} - 1
```

The `-lambda k` term is the compensator that keeps the discounted price a
martingale. Conditional on `N_T = n` the jump sum is itself `N(n mu_J, n sigma_J^2)`,
so the whole thing is one extra normal draw — no per-jump loop.

Conditional on `n` jumps the terminal law is again lognormal, so the price is a
Poisson-weighted sum of Black-Scholes prices. That series is implemented as
`merton_call_price` and used as the oracle: the simulator matches it to within
0.04 standard errors at 2M paths.

### Heston stochastic volatility

```
dS = r S dt + sqrt(v) S dW1
dv = kappa (theta - v) dt + xi sqrt(v) dW2,     corr(dW1, dW2) = rho
```

`kappa` is mean-reversion speed, `theta` the long-run variance, `xi` the
vol-of-vol. The Feller condition `2 kappa theta > xi^2` keeps variance positive
in the continuous SDE, but Euler steps can still go negative regardless.

This uses the **full truncation** scheme: the possibly-negative variance stays in
the state, and `max(v, 0)` is applied wherever variance is *read*. Among the
simple fixes (reflection, absorption, truncation) this has the smallest
discretisation bias.

Tested by: `xi = 0, v0 = theta` recovering flat-vol GBM statistically, the
martingale property, `rho < 0` producing negative log-price skew and `rho > 0`
flipping it, strict positivity of prices, and variance mean-reverting to `theta`
from `v0 = 0.16` over five years.

---

## 6. European pricing and convergence

![convergence](figures/convergence.png)

| paths | MC price | standard error |
|---|---|---|
| 1,000 | 9.8523 | 0.4552 |
| 10,000 | 10.5456 | 0.1470 |
| 100,000 | 10.5287 | 0.0468 |
| 1,000,000 | 10.4537 | 0.0147 |

The error falls by a factor of ten for every hundredfold increase in paths — the
signature of the `O(N^{-1/2})` rate. The right-hand panel plots the observed
standard error against an `N^{-1/2}` reference line.

Coverage is checked directly rather than assumed: across 100 seeds, the reported
95% confidence interval contains the true Black-Scholes price in 88–100 runs.

---

## 7. Variance reduction

![variance reduction](figures/variance_reduction.png)

| estimator | price | standard error | variance removed | efficiency |
|---|---|---|---|---|
| naive | 10.4513 | 0.02082 | — | 1.00× |
| antithetic | 10.4409 | 0.01466 | 50.4% | 2.20× |
| control variate | 10.4466 | 0.00834 | 83.9% | 5.97× |

The control variate reaches in 500,000 paths an accuracy naive sampling would
need roughly 3.1 million paths to match.

### Antithetic variates

For each draw `Z`, also use `-Z`:

```
Var((X1 + X2)/2) = [Var(X1) + Var(X2) + 2 Cov(X1, X2)] / 4
```

Independent draws give `Var(X)/2`; a negative covariance beats that. `S_T` is
monotone increasing in `Z` and a vanilla payoff is monotone in `S_T`, so the
paired payoffs are negatively correlated. **The precondition is monotonicity** —
for a straddle or a butterfly, antithetic sampling can *increase* variance.

### Control variates

With `Y` of known mean and correlated with the payoff `X`:

```
X_cv = X - c (Y - E[Y])
```

Unbiased for any `c`. Minimising `Var(X) - 2c Cov(X,Y) + c^2 Var(Y)` over `c`
gives

```
c* = Cov(X, Y) / Var(Y)        and        Var(X_cv) = Var(X)(1 - rho^2)
```

which is exactly the OLS slope of `X` on `Y` — geometrically, projecting the
payoff onto the control and keeping only the residual. Here `Y = S_T`, whose mean
`S0 e^{rT}` is known exactly.

`c` is fitted on a **disjoint pilot sample** (10% of paths by default) and applied
to the main sample. Estimating it from the same draws it corrects would correlate
the coefficient with the noise and bias the estimator. Deep in the money the
payoff becomes exactly linear in the control, `rho -> 1`, and the residual
variance collapses — the test suite asserts the standard error falls below 2%
of the naive estimator's at `S0 = 300`.

### Why efficiency, not variance reduction

Variance-per-path flatters an estimator that costs more per path. Antithetic
draws half the normals for the same path count; the control variate pays for a
pilot run and a covariance estimate. The honest metric is `variance × runtime`
against naive, so an estimator scores above 1.0 only if it reaches a given error
in less wall-clock time. At this scale the two measures nearly agree because
random number generation is not the bottleneck — but the right metric is the one
that stays right when it isn't.

---

## 8. Greeks

| Greek | method | Monte Carlo | closed form | error |
|---|---|---|---|---|
| delta | finite difference | 0.63715 ± 0.00041 | 0.63683 | 0.78 se |
| delta | pathwise | 0.63637 ± 0.00041 | 0.63683 | 1.13 se |
| vega | finite difference | 37.5612 ± 0.0536 | 37.5240 | 0.69 se |
| vega | pathwise | 37.4281 ± 0.0534 | 37.5240 | 1.80 se |
| gamma | finite difference | 0.018813 ± 0.000079 | 0.018762 | 0.67 se |

### Common random numbers — the critical detail

The central difference `[C(S0+h) - C(S0-h)] / 2h` must price both legs on the
**same** normal draws:

```
Var[(X+ - X-) / 2h] = [Var(X+) + Var(X-) - 2 Cov(X+, X-)] / 4h^2
```

With independent draws the covariance is zero and the variance explodes like
`1/h^2`. With shared draws the two prices are almost perfectly correlated, the
differences nearly cancel, and the variance stays bounded. **Measured here:
dropping common random numbers inflates delta's standard error by roughly
1,800×.** The estimator is exposed as `common_random_numbers=True` with a test
that turns it off purely to demonstrate the failure.

### Pathwise estimators

Differentiate the payoff along each path instead of bumping. Since `S_T` is
linear in `S0`:

```
delta = e^{-rT} E[1{S_T > K} S_T / S0]                       (call)
vega  = e^{-rT} E[1{S_T > K} S_T (sqrt(T) Z - sigma T)]      (call)
```

No bump means no bias to trade off against variance. Validity requires the payoff
to be Lipschitz and differentiable almost everywhere — true for vanillas, false
for digitals, whose payoff is a step function.

As the bump shrinks, the CRN difference quotient converges to the pathwise
derivative, so the two estimators have nearly identical standard errors — the
test asserts they agree within 5%. The pathwise estimator's real advantage over
finite differences is against a *non*-CRN implementation, where it wins by more
than two orders of magnitude.

**Gamma has no pathwise estimator.** The first derivative already contains an
indicator function, whose derivative is a Dirac delta. It must use finite
differences, and because the second difference divides by `h^2` it needs a
noticeably wider bump than delta — `h = 1.0` here versus `h = 0.01`.

---

## 9. American options

| method | price |
|---|---|
| Longstaff-Schwartz (500k paths, 50 steps) | 6.0846 ± 0.0101 |
| binomial tree (4,000 steps) | 6.0902 |
| European put (Black-Scholes) | 5.5735 |

Early exercise premium **0.5111**.

### Binomial tree (the American oracle)

Cox-Ross-Rubinstein: `u = e^{sigma sqrt(dt)}`, `d = 1/u`, and the risk-neutral
probability `p = (e^{r dt} - d) / (u - d)`, chosen so the discounted expected
price equals today's. `u d = 1` makes the tree recombining, so `n` steps give
`n+1` nodes rather than `2^n`. Backward induction takes
`max(continuation, intrinsic)` at each node for American exercise.

Deterministic and accurate to well under a cent at 4,000 steps, so it serves as
ground truth where no closed form exists.

### Longstaff-Schwartz

American pricing is an optimal-stopping problem: at each exercise date, compare
immediate value against continuation value. The difficulty is that continuation
value is a conditional expectation you cannot observe forward in a simulation.

LSM estimates it by regression. Working backwards from maturity, realised
discounted future cashflows are regressed on polynomials in `S/K` (regressing on
the normalised spot keeps the Vandermonde matrix well conditioned), and exercise
happens wherever immediate value beats the fitted continuation. **Only
in-the-money paths enter the regression** — out-of-the-money paths carry no
exercise decision and would distort the fit where it matters.

**LSM is a lower bound.** Exercise is restricted to the simulation grid and the
exercise rule is estimated rather than optimal; both lose value. It tightens as
the grid refines, which the test suite demonstrates directly: a deep in-the-money
put worth exactly 60 prices at 59.79 with 25 steps and 59.95 with 100.

The American call is tested to equal the European call within one standard error.
Early exercise pays `K` sooner, losing interest on it, and destroys remaining time
value — formally `C >= S0 - K e^{-rT} > S0 - K`, so the call is always worth more
alive than exercised. This breaks with dividends.

---

## 10. Alternative dynamics and volatility smiles

![smiles](figures/volatility_smiles.png)

| strike | 80 | 90 | 100 | 110 | 120 |
|---|---|---|---|---|---|
| GBM | 0.1963 | 0.1981 | 0.1987 | 0.1989 | 0.1991 |
| Merton jumps | 0.2489 | 0.2426 | 0.2387 | 0.2365 | 0.2353 |
| Heston | 0.2447 | 0.2163 | 0.1887 | 0.1634 | 0.1448 |

GBM is flat by construction: one volatility goes in, the same one comes back out
at every strike. Merton's downward jumps fatten the left tail and lift low-strike
implied volatility, but the skew is shallow and decays with maturity — jump risk
is a short-dated phenomenon. Heston's negative price-variance correlation (the
leverage effect: prices fall, volatility rises) produces the steep skew seen in
equity index markets, and because it comes from a second stochastic factor it
persists to longer maturities.

Implied volatility is recovered by Brent's method on `[0, 5]`. The Black-Scholes
price is strictly increasing in `sigma`, so the root is unique and bracketing is
safe; Newton's method can leave the bracket where vega is near zero. Where vega
genuinely vanishes — deep in the money at low volatility — sigma is **not
identifiable** from the price at any useful precision, so the test suite skips
exactly those points rather than loosening its tolerance.

---

## 11. The hedging backtest

![hedging backtest](figures/hedging_backtest.png)

### Construction

Each cycle sells a one-month at-the-money straddle on the index, hedges it with
the engine's delta, and holds to expiry. 118 non-overlapping cycles,
September 2016 to September 2026. Three real daily series from FRED, cached in
`data/`:

| series | meaning |
|---|---|
| `SP500` | S&P 500 index level — the spot path |
| `VIXCLS` | VIX — the market's 30-day implied volatility |
| `DGS1MO` | 1-month Treasury yield — the discount rate |

Daily P&L is the mark-to-market change on the option position plus the return on
shares held into that day:

```
pnl_t = sign (V_t - V_{t-1}) + h_{t-1} (S_t - S_{t-1})
```

where `h` cancels the option delta at each rehedge and is held fixed in between.
Everything is divided by the strike, so results are fractions of notional and
comparable across a decade of index levels.

### Results

| scenario | ann. return | ann. vol | Sharpe | max drawdown |
|---|---|---|---|---|
| frictionless, daily hedge | +11.2% | 5.2% | 2.14 | −5.7% |
| IV −1.5 pts, 1bp cost, daily | +6.8% | 5.2% | 1.31 | −6.3% |
| IV −1.5 pts, 1bp cost, weekly | +7.6% | 7.3% | 1.04 | −14.3% |
| IV −3 pts, 1bp cost, daily | +2.8% | 5.2% | 0.54 | −10.6% |
| unhedged, IV −1.5 pts | +4.1% | 11.5% | 0.36 | −32.0% |

### What the experiment shows

**Hedging creates the Sharpe, not the premium.** Unhedged, the same
short-volatility position earns 0.36 and draws down 32%. The premium is collected
either way; delta hedging strips out the directional variance around it, leaving
the variance bet. For a delta-hedged short option the P&L over `dt` is
approximately

```
-(1/2) Gamma S^2 (sigma_realised^2 - sigma_implied^2) dt
```

You are short gamma: you profit when realised volatility comes in below the
implied volatility you sold.

**The edge is dominated by the volatility you sell at.** The VIX is a strip across
strikes and sits above the at-the-money implied volatility a straddle actually
trades at. Selling 1.5 points below the VIX cuts the Sharpe from 2.14 to 1.31;
three points takes it to 0.54. Without real option chains this cannot be pinned
down, so it is reported as a sensitivity rather than buried in a headline number.

**Rehedging daily beats weekly even after costs.** Weekly earns slightly more
gross but carries 40% more volatility and more than twice the drawdown. At 1bp
per rehedge, transaction costs are small next to the gamma risk of leaving the
position unhedged.

**The premium is real and it is compensation for tail risk.** Mean entry implied
volatility is 17.3% against 15.1% subsequently realised — a 2.2 point variance
risk premium. The worst cycle opened 19 February 2020 at an implied 12.9% and
realised 84.3%; second worst was February 2018. Short volatility pays steadily
and loses violently, and the P&L path shows it.

### What is real and what is constructed

The spot path, implied volatility and discount rate are **real observations**.
The option prices are **not**: historical chains are not freely available, so each
straddle is priced by Black-Scholes at the VIX. The premium was never a traded
price and the daily marks are model marks. No bid-ask, no skew, no term-structure
interpolation. The `iv_offset` sensitivity exists precisely because of this.

The engine and the backtest are connected by more than reuse: a test hedges a
full cycle with **Monte Carlo pathwise deltas** instead of closed-form ones and
asserts the resulting P&L agrees to within 5e-4.

---

## 12. Validation methodology

**Two oracles, tested first.** `black_scholes.py` and `binomial.py` contain no
randomness. They are pinned to exact textbook values (Hull's canonical example:
call 10.4506, put 5.5735) and to identities that must hold for any parameters —
put-call parity, no-arbitrage price bounds, monotonicity in spot and volatility,
deep-ITM convergence to intrinsic value, and the degenerate `sigma=0` and `T=0`
cases. Their Greeks are cross-checked by finite-differencing their own price
functions, so a later disagreement between a Monte Carlo estimator and the closed
form can only be the estimator's fault.

**Statistical assertions in standard errors, not tolerances.** A tolerance is a
number you tune until the test passes. A standard error is computed from the
sample and makes the assertion a statistical statement — a Monte Carlo price must
fall within three standard errors of its reference. Coverage is verified
directly: across 100 seeds the 95% interval contains the true price in 88–100
runs.

**Invariants that must hold exactly.** `sigma=0` collapses every path onto
`S0 e^{rT}` to floating-point precision. `T=0` returns spot. The discounted price
is a martingale. Put-call parity holds to `1e-10`. The Itô drift is tested
separately from `E[S_T]`, because a sign error in `-sigma^2/2` fails the two
tests differently.

**Metamorphic tests where no single right answer exists.** Merton with
`lambda = 0` must reproduce GBM. Heston with `xi = 0` and `v0 = theta` must
reproduce flat-vol GBM. These check a relation between outputs rather than any
one output.

### Test suite

| module | test functions |
|---|---|
| `test_black_scholes.py` | 29 |
| `test_jump_and_heston.py` | 19 |
| `test_greeks.py` | 17 |
| `test_validation.py` | 17 |
| `test_simulation.py` | 16 |
| `test_pricing_european.py` | 15 |
| `test_variance_reduction.py` | 15 |
| `test_backtest.py` | 14 |
| `test_binomial.py` | 10 |
| `test_pricing_american.py` | 7 |
| `test_data.py` | 4 |
| `test_import.py` | 1 |

164 test functions expand to **202 tests** under parametrisation. Coverage is 98%
on non-plotting modules; `reporting.py` is excluded so the figure describes the
mathematics rather than the plotting code.

---

## 13. Running it

```bash
pip install -r requirements.txt

pytest                              # 202 tests
pytest --cov                        # with coverage
python scripts/generate_report.py   # regenerate figures/ and all printed tables
```

The notebook at `notebooks/results_report.ipynb` reproduces every figure and
table with stored outputs, so it renders without being run.

### Layout

```
src/mcpricer/
  black_scholes.py        closed-form prices, Greeks, Merton series, implied vol
  binomial.py             Cox-Ross-Rubinstein tree
  simulation.py           GBM, Merton jump-diffusion, Heston
  payoffs.py              vanilla payoff functions
  pricing.py              European MC, convergence study, Longstaff-Schwartz
  variance_reduction.py   antithetic and control variate estimators
  greeks.py               finite-difference and pathwise estimators
  data.py                 cached FRED market data
  backtest.py             delta-hedged straddle backtest
  reporting.py            figures and summary tables
data/                     SP500, VIX and 1-month T-bill daily closes
tests/                    one module per source module
notebooks/                executed results report
scripts/                  figure and table regeneration
```

---

## 14. Limitations and next steps

**No historical option chains.** The single largest limitation. Straddles are
constructed at the VIX rather than at traded quotes, so there is no bid-ask, no
skew, and no term-structure interpolation. The measured edge sits inside the
uncertainty of an input that cannot be observed from free data.

**118 cycles is a small sample** for a strategy whose risk is concentrated in the
tail, and two cycles dominate the loss distribution. Cycles are non-overlapping
and start on a fixed calendar offset.

**Heston and Merton parameters are illustrative, not calibrated.** They are
plausible textbook values chosen to demonstrate the models' qualitative
behaviour, not fitted to any surface.

**Quasi-random sequences are not implemented.** Sobol sequences with a Brownian
bridge construction would improve on the `O(N^{-1/2})` rate for low effective
dimension; only antithetic and control variates are built here.

**The natural next step is calibration.** Fitting Heston to a real option surface
would both extend the modelling half and supply the at-the-money implied
volatility the backtest currently has to approximate with the VIX — closing the
loop on the project's biggest open question.
