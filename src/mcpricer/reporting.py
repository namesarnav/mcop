"""Figures and summary tables for the results writeup.

Kept out of the priced modules so that coverage targets apply to the maths and
not to plotting code.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcpricer.black_scholes import (
    bs_call_delta,
    bs_call_price,
    bs_gamma,
    bs_vega,
    implied_volatility_smile,
)
from mcpricer.greeks import (
    mc_delta_finite_difference,
    mc_delta_pathwise,
    mc_gamma_finite_difference,
    mc_vega_finite_difference,
    mc_vega_pathwise,
)
from mcpricer.payoffs import call_payoff
from mcpricer.pricing import convergence_study
from mcpricer.simulation import (
    simulate_gbm_terminal,
    simulate_heston_paths,
    simulate_merton_terminal,
)
from mcpricer.variance_reduction import compare_estimators

BASE = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)

__all__ = [
    "plot_convergence",
    "plot_variance_reduction",
    "greeks_table",
    "plot_volatility_smiles",
]


def plot_convergence(path: Path, seed: int = 20260905) -> dict[str, np.ndarray]:
    """MC price with a 95% band against the closed-form price, versus N."""
    counts = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    study = convergence_study(**BASE, path_counts=counts, seed=seed)
    truth = bs_call_price(**BASE)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    band = 1.96 * study["standard_error"]
    axes[0].fill_between(
        study["n_paths"], study["price"] - band, study["price"] + band, alpha=0.25
    )
    axes[0].plot(study["n_paths"], study["price"], marker="o", label="Monte Carlo")
    axes[0].axhline(truth, color="black", linestyle="--", label="Black-Scholes")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("paths")
    axes[0].set_ylabel("call price")
    axes[0].set_title("Convergence with 95% interval")
    axes[0].legend()

    axes[1].loglog(study["n_paths"], study["standard_error"], marker="o", label="observed")
    reference = study["standard_error"][0] * np.sqrt(study["n_paths"][0] / study["n_paths"])
    axes[1].loglog(study["n_paths"], reference, linestyle="--", label=r"$N^{-1/2}$")
    axes[1].set_xlabel("paths")
    axes[1].set_ylabel("standard error")
    axes[1].set_title("Error decay")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return study


def plot_variance_reduction(path: Path, n_paths: int = 500_000, seed: int = 7) -> list[dict]:
    """Standard error of each estimator at a matched path count."""
    rows = compare_estimators(**BASE, n_paths=n_paths, seed=seed)
    figure, ax = plt.subplots(figsize=(6, 4))
    ax.bar([row["method"] for row in rows], [row["standard_error"] for row in rows])
    ax.set_ylabel("standard error")
    ax.set_title(f"Standard error at N = {n_paths:,}")
    for index, row in enumerate(rows):
        ax.text(
            index,
            row["standard_error"],
            f"{row['variance_reduction_pct']:.0f}%",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return rows


def greeks_table(n_paths: int = 2_000_000, seed: int = 11) -> list[dict[str, object]]:
    """Every Monte Carlo Greek estimator against its closed-form value."""
    estimators = [
        ("delta", "finite difference", mc_delta_finite_difference(**BASE, n_paths=n_paths, seed=seed), bs_call_delta(**BASE)),
        ("delta", "pathwise", mc_delta_pathwise(**BASE, n_paths=n_paths, seed=seed + 1), bs_call_delta(**BASE)),
        ("vega", "finite difference", mc_vega_finite_difference(**BASE, n_paths=n_paths, seed=seed + 2), bs_vega(**BASE)),
        ("vega", "pathwise", mc_vega_pathwise(**BASE, n_paths=n_paths, seed=seed + 3), bs_vega(**BASE)),
        ("gamma", "finite difference", mc_gamma_finite_difference(**BASE, n_paths=n_paths, seed=seed + 4), bs_gamma(**BASE)),
    ]
    return [
        {
            "greek": greek,
            "method": method,
            "estimate": estimate.value,
            "standard_error": estimate.standard_error,
            "closed_form": truth,
            "error_in_standard_errors": abs(estimate.value - truth) / estimate.standard_error,
        }
        for greek, method, estimate, truth in estimators
    ]


def plot_volatility_smiles(path: Path, n_paths: int = 500_000) -> dict[str, np.ndarray]:
    """Implied volatility by strike under GBM, Merton and Heston."""
    strikes = np.linspace(80.0, 120.0, 9)
    discount = np.exp(-BASE["r"] * BASE["T"])

    terminals = {
        "GBM": simulate_gbm_terminal(
            S0=BASE["S0"], r=BASE["r"], sigma=BASE["sigma"], T=BASE["T"],
            n_paths=n_paths, seed=1,
        ),
        "Merton jumps": simulate_merton_terminal(
            S0=BASE["S0"], r=BASE["r"], sigma=BASE["sigma"], T=BASE["T"],
            n_paths=n_paths, seed=2,
            jump_intensity=0.75, jump_mean=-0.05, jump_std=0.15,
        ),
        "Heston": simulate_heston_paths(
            S0=BASE["S0"], r=BASE["r"], T=BASE["T"], n_paths=n_paths, n_steps=100,
            seed=3, v0=0.04, kappa=2.0, theta=0.04, xi=0.6, rho=-0.7,
        )[:, -1],
    }

    smiles = {}
    figure, ax = plt.subplots(figsize=(6, 4))
    for label, terminal in terminals.items():
        prices = np.array([discount * call_payoff(terminal, k).mean() for k in strikes])
        smiles[label] = implied_volatility_smile(
            prices, BASE["S0"], strikes, BASE["r"], BASE["T"]
        )
        ax.plot(strikes, smiles[label], marker="o", label=label)
    ax.set_xlabel("strike")
    ax.set_ylabel("implied volatility")
    ax.set_title("Implied volatility by strike, T = 1")
    ax.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    smiles["strikes"] = strikes
    return smiles
