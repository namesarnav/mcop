# Monte Carlo Options Pricer

A Monte Carlo engine for pricing European and American options, with variance
reduction, Greeks estimation, and alternative asset dynamics. Every stochastic
result is validated against an independent deterministic reference — closed-form
Black-Scholes, Merton's series solution, or a binomial tree — so correctness is
demonstrated rather than asserted.

98% test coverage on the pricing and simulation modules. All figures and tables
below are reproducible from fixed seeds via `python scripts/generate_report.py`.

## Results

Reference contract throughout: `S0 = K = 100`, `r = 5%`, `sigma = 20%`, `T = 1`.
Black-Scholes call price **10.4506**.

### Convergence

![convergence](figures/convergence.png)

| paths | MC price | standard error |
|---|---|---|
| 1,000 | 9.8523 | 0.4552 |
| 10,000 | 10.5456 | 0.1470 |
| 100,000 | 10.5287 | 0.0468 |
| 1,000,000 | 10.4537 | 0.0147 |

The error falls by a factor of ten for every hundredfold increase in paths, the
signature of the `O(N^-1/2)` rate.

### Variance reduction (N = 500,000)

![variance reduction](figures/variance_reduction.png)

| estimator | price | standard error | variance removed |
|---|---|---|---|
| naive | 10.4513 | 0.02082 | — |
| antithetic | 10.4409 | 0.01466 | 50.4% |
| control variate | 10.4466 | 0.00834 | 83.9% |

The control variate reaches in 500,000 paths an accuracy that naive sampling
would need roughly 3.1 million paths to match.

### Greeks (N = 2,000,000)

| Greek | method | Monte Carlo | closed form | error |
|---|---|---|---|---|
| delta | finite difference | 0.63715 ± 0.00041 | 0.63683 | 0.78 se |
| delta | pathwise | 0.63637 ± 0.00041 | 0.63683 | 1.13 se |
| vega | finite difference | 37.5612 ± 0.0536 | 37.5240 | 0.69 se |
| vega | pathwise | 37.4281 ± 0.0534 | 37.5240 | 1.80 se |
| gamma | finite difference | 0.018813 ± 0.000079 | 0.018762 | 0.67 se |

### American put

| method | price |
|---|---|
| Longstaff-Schwartz (500k paths, 50 steps) | 6.0846 ± 0.0101 |
| binomial tree (4,000 steps) | 6.0902 |
| European put (Black-Scholes) | 5.5735 |

Early exercise premium **0.5111**. The American call price matches the European
call to within one standard error, as it must without dividends.

### Volatility smiles

![smiles](figures/volatility_smiles.png)

| strike | 80 | 90 | 100 | 110 | 120 |
|---|---|---|---|---|---|
| GBM | 0.1963 | 0.1981 | 0.1987 | 0.1989 | 0.1991 |
| Merton jumps | 0.2489 | 0.2426 | 0.2387 | 0.2365 | 0.2353 |
| Heston | 0.2447 | 0.2163 | 0.1887 | 0.1634 | 0.1448 |

GBM is flat by construction: one volatility goes in, the same one comes back
out at every strike. Merton's downward jumps fatten the left tail and lift
low-strike implied volatility. Heston's negative price-variance correlation
produces the steep skew seen in equity index markets.

## The maths

**Risk-neutral pricing.** Under no arbitrage there exists a measure under which
every discounted tradable price is a martingale. The option's price today is
therefore the discounted expectation of its payoff under that measure,
`C = e^{-rT} E[payoff(S_T)]`, with the asset drifting at `r` rather than at its
real-world return. Monte Carlo estimates that expectation by sampling.

**Simulation.** Applying Itô's lemma to `log S_t` gives the exact solution

```
S_T = S0 exp[(r - sigma^2/2) T + sigma sqrt(T) Z],   Z ~ N(0,1)
```

Sampling in log space means no discretisation error and no possibility of a
negative price. For European payoffs no intermediate time steps are needed at
all; the full grid is only simulated for early exercise and stochastic
volatility.

**Convergence.** The estimator's standard error is
`e^{-rT} sigma_payoff / sqrt(N)`. Halving the error costs four times the paths;
one more decimal digit costs a hundred times. That cost curve is why variance
reduction matters more than brute force.

**Antithetic variates.** `S_T` is monotone in `Z`, so the payoffs from `Z` and
`-Z` are negatively correlated and their average has lower variance than two
independent draws. The technique relies on that monotonicity, and would help
much less for a payoff that is not monotone in the underlying.

**Control variates.** For a control `Y` with known mean, `X - c(Y - E[Y])` is
unbiased for any `c`, and its variance is minimised at `c = Cov(X,Y)/Var(Y)` —
the least-squares slope of `X` on `Y`, which projects out the component of the
payoff that `Y` explains. Here `Y = S_T`, whose mean `S0 e^{rT}` is known
exactly. The coefficient is fitted on a pilot sample and applied to a disjoint
main sample, so it is independent of the draws it corrects.

**Greeks.** Finite-difference estimators must price both legs of the bump on the
*same* normal draws. Otherwise the difference of two independently noisy prices
is divided by a small bump and the sampling noise is amplified by `1/h`; in this
project's tests, dropping common random numbers inflates delta's standard error
by a factor of roughly 1,800. The pathwise estimator differentiates the payoff
along each path instead — `dC/dS0 = e^{-rT} E[1{S_T > K} S_T / S0]` — which
removes the bump entirely. Gamma has no pathwise estimator, since the first
derivative already contains an indicator function.

**American options.** Longstaff-Schwartz works backwards from maturity,
regressing realised discounted future cashflows on a polynomial in the current
spot to estimate the continuation value, and exercising where immediate value
exceeds it. Only in-the-money paths enter the regression. Because exercise is
restricted to the simulation grid and the exercise rule is estimated, LSM is a
lower bound that tightens as the grid is refined.

## Validation methodology

Two oracles contain no randomness and are tested against exact values first:
`black_scholes.py` (textbook prices, put-call parity, degenerate cases,
no-arbitrage bounds, Greeks cross-checked by finite-differencing the price
function) and `binomial.py` (convergence to Black-Scholes for European exercise).
Everything stochastic is then tested against them.

Statistical assertions are stated in standard errors computed from the sample
itself, never as hand-tuned tolerances — a Monte Carlo price must fall within
three standard errors of its closed-form reference. Coverage is checked
directly: across 100 seeds, the reported 95% confidence interval contains the
true price in 88-100 runs.

Simulators are additionally pinned by properties that must hold exactly:
`sigma = 0` collapses every path onto `S0 e^{rT}`; `T = 0` returns spot; the
discounted price is a martingale; the Itô drift correction is tested separately
from `E[S_T]`, since a sign error there fails the two tests differently.

## Layout

```
src/mcpricer/
  black_scholes.py     closed-form prices, Greeks, Merton series, implied vol
  binomial.py          Cox-Ross-Rubinstein tree (American reference)
  simulation.py        GBM, Merton jump-diffusion, Heston
  payoffs.py           vanilla payoff functions
  pricing.py           European MC, convergence study, Longstaff-Schwartz
  variance_reduction.py  antithetic and control variate estimators
  greeks.py            finite-difference and pathwise estimators
  reporting.py         figures and summary tables
tests/                 one module per source module
notebooks/             results writeup
scripts/               figure and table regeneration
```

## Running

```bash
pip install -r requirements.txt
pytest                       # full suite
pytest --cov                 # with coverage
python scripts/generate_report.py   # regenerate figures/ and printed tables
```
