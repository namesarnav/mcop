"""Regenerate every figure and table in the README.

Usage: python scripts/generate_report.py
"""

from __future__ import annotations

from pathlib import Path

from mcpricer.binomial import binomial_american_put_price
from mcpricer.black_scholes import bs_call_price, bs_put_price
from mcpricer.pricing import lsm_american_price
from mcpricer.reporting import (
    greeks_table,
    plot_convergence,
    plot_variance_reduction,
    plot_volatility_smiles,
)

BASE = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
FIGURES = Path(__file__).resolve().parents[1] / "figures"


def main() -> None:
    FIGURES.mkdir(exist_ok=True)

    study = plot_convergence(FIGURES / "convergence.png")
    print("convergence (Black-Scholes = %.4f)" % bs_call_price(**BASE))
    for n, price, se in zip(study["n_paths"], study["price"], study["standard_error"]):
        print(f"  N={n:>9,}  price={price:8.4f}  se={se:7.4f}")

    print("\nvariance reduction at N = 500,000")
    for row in plot_variance_reduction(FIGURES / "variance_reduction.png"):
        print(
            f"  {row['method']:<16} price={row['price']:.4f} "
            f"se={row['standard_error']:.5f} "
            f"variance reduction={row['variance_reduction_pct']:6.2f}%"
        )

    print("\ngreeks")
    for row in greeks_table():
        print(
            f"  {row['greek']:<6} {row['method']:<18} "
            f"mc={row['estimate']:9.5f} +/- {row['standard_error']:.5f}  "
            f"closed form={row['closed_form']:9.5f}  "
            f"({row['error_in_standard_errors']:.2f} se)"
        )

    lsm = lsm_american_price(**BASE, n_paths=500_000, n_steps=50, seed=5)
    tree = binomial_american_put_price(**BASE, n_steps=4000)
    print("\namerican put")
    print(f"  LSM           {lsm.price:.4f} +/- {lsm.standard_error:.4f}")
    print(f"  binomial tree {tree:.4f}")
    print(f"  european put  {bs_put_price(**BASE):.4f}")
    print(f"  early exercise premium {lsm.price - bs_put_price(**BASE):.4f}")

    smiles = plot_volatility_smiles(FIGURES / "volatility_smiles.png")
    print("\nimplied volatility smiles")
    print(f"  {'strike':<13}" + "  ".join(f"{k:6.1f}" for k in smiles["strikes"]))
    for label in ("GBM", "Merton jumps", "Heston"):
        print(f"  {label:<13}" + "  ".join(f"{v:6.4f}" for v in smiles[label]))


if __name__ == "__main__":
    main()
