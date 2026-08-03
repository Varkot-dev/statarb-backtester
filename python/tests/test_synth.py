"""Tests for synthetic data generation.

The critical property is that the generator produces series whose true
parameters match what was requested — otherwise the ground-truth tests that
validate the estimators are checking against the wrong answer, and every
downstream correctness claim rests on nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.tsa.stattools import adfuller

from statarb.cointegration import estimate_half_life, ols_hedge_ratio
from statarb.synth import CointegratedSpec, generate_cointegrated, generate_independent


def test_spec_rejects_invalid_parameters():
    """Invalid specs fail at construction, not at generation time."""
    with pytest.raises(ValueError, match="n_bars"):
        CointegratedSpec(n_bars=1, beta=1.0, half_life=10.0, sigma_spread=0.02, sigma_common=0.01)
    with pytest.raises(ValueError, match="half_life"):
        CointegratedSpec(n_bars=100, beta=1.0, half_life=0.0, sigma_spread=0.02, sigma_common=0.01)
    with pytest.raises(ValueError, match="sigma_spread"):
        CointegratedSpec(n_bars=100, beta=1.0, half_life=10.0, sigma_spread=-0.02, sigma_common=0.01)
    with pytest.raises(ValueError, match="positive"):
        CointegratedSpec(
            n_bars=100, beta=1.0, half_life=10.0, sigma_spread=0.02,
            sigma_common=0.01, price_a0=0.0,
        )


def test_phi_matches_the_requested_half_life():
    """phi = exp(-ln2 / h) must decay to exactly half after h bars."""
    for half_life in (5.0, 15.0, 45.0):
        spec = CointegratedSpec(
            n_bars=100, beta=1.0, half_life=half_life, sigma_spread=0.02, sigma_common=0.01
        )
        assert spec.phi**half_life == pytest.approx(0.5, rel=1e-12)


def test_innovation_sd_gives_the_requested_unconditional_sd():
    """A stationary AR(1) has variance sigma_e^2 / (1 - phi^2)."""
    spec = CointegratedSpec(
        n_bars=100, beta=1.0, half_life=20.0, sigma_spread=0.03, sigma_common=0.01
    )
    implied = spec.innovation_sd / np.sqrt(1.0 - spec.phi**2)
    assert implied == pytest.approx(spec.sigma_spread, rel=1e-12)


def test_generation_is_deterministic_given_a_seed():
    """Same seed, byte-identical output. This is what makes `make backtest` reproducible."""
    spec = CointegratedSpec(
        n_bars=300, beta=1.2, half_life=15.0, sigma_spread=0.03, sigma_common=0.01
    )
    a = generate_cointegrated(spec, seed=42)
    b = generate_cointegrated(spec, seed=42)
    np.testing.assert_array_equal(a["price_a"].to_numpy(), b["price_a"].to_numpy())
    np.testing.assert_array_equal(a["price_b"].to_numpy(), b["price_b"].to_numpy())


def test_different_seeds_give_different_series():
    """Guard against a seed that is accidentally ignored."""
    spec = CointegratedSpec(
        n_bars=300, beta=1.2, half_life=15.0, sigma_spread=0.03, sigma_common=0.01
    )
    a = generate_cointegrated(spec, seed=1)
    b = generate_cointegrated(spec, seed=2)
    assert not np.allclose(a["price_a"].to_numpy(), b["price_a"].to_numpy())


def test_prices_are_positive_and_finite():
    """The engine takes logs, so non-positive prices would be fatal downstream."""
    spec = CointegratedSpec(
        n_bars=500, beta=1.1, half_life=12.0, sigma_spread=0.04, sigma_common=0.02
    )
    df = generate_cointegrated(spec, seed=7)
    for column in ("price_a", "price_b"):
        values = df[column].to_numpy()
        assert np.all(values > 0), f"{column} has non-positive values"
        assert np.all(np.isfinite(values)), f"{column} has non-finite values"


def test_dates_are_strictly_increasing():
    """The engine's CSV loader rejects non-chronological data."""
    spec = CointegratedSpec(
        n_bars=200, beta=1.0, half_life=10.0, sigma_spread=0.02, sigma_common=0.01
    )
    dates = generate_cointegrated(spec, seed=3)["date"]
    assert dates.is_monotonic_increasing
    assert not dates.duplicated().any()


@pytest.mark.parametrize("true_beta", [0.7, 1.0, 1.35])
def test_ground_truth_beta_is_recoverable(true_beta: float):
    """The generator's declared beta must be what OLS recovers.

    This is the test that licenses using synthetic data as ground truth. If it
    failed, every "the estimator recovers the true parameter" claim elsewhere
    would be comparing against a number the generator never actually used.
    """
    spec = CointegratedSpec(
        n_bars=3000,
        beta=true_beta,
        half_life=15.0,
        sigma_spread=0.02,
        sigma_common=0.02,
    )
    df = generate_cointegrated(spec, seed=11)
    _, beta_hat = ols_hedge_ratio(
        np.log(df["price_a"].to_numpy()), np.log(df["price_b"].to_numpy())
    )
    assert beta_hat == pytest.approx(true_beta, rel=0.05)


@pytest.mark.parametrize("true_half_life", [8.0, 20.0, 50.0])
def test_ground_truth_half_life_is_recoverable(true_half_life: float):
    """The generator's declared half-life must be what the OU regression recovers.

    Averaged across seeds, not pinned to one draw. The half-life estimator has
    substantial sampling variation on a single path — at a true half-life of 50
    and 4000 bars, individual estimates range roughly 33 to 59 (sd ≈ 8) — so a
    single-seed assertion with a tight tolerance would be testing which seed
    was chosen rather than whether the estimator works.

    Averaging over 8 independent paths is the meaningful claim: the estimator
    is centred on the truth. The separate convergence test below confirms it is
    consistent (the error shrinks with sample size), which distinguishes
    sampling noise from genuine bias.
    """
    estimates = [
        estimate_half_life(
            generate_cointegrated(
                CointegratedSpec(
                    n_bars=4000,
                    beta=1.0,
                    half_life=true_half_life,
                    sigma_spread=0.03,
                    sigma_common=0.01,
                ),
                seed=seed,
            )["true_spread"].to_numpy()
        )
        for seed in range(8)
    ]
    assert all(np.isfinite(e) for e in estimates), "every path should be mean-reverting"
    assert float(np.mean(estimates)) == pytest.approx(true_half_life, rel=0.20)


def test_half_life_estimator_is_consistent():
    """The estimation error must shrink as the sample grows.

    This is what separates sampling noise from bias. A biased estimator would
    converge to the wrong number no matter how much data it got; a consistent
    one converges to the truth. Together with the averaged test above, this
    establishes that the ground-truth machinery is sound — which every other
    "the estimator recovers the true parameter" claim depends on.
    """
    true_half_life = 20.0
    errors = []
    for n_bars in (2000, 20000):
        estimates = [
            estimate_half_life(
                generate_cointegrated(
                    CointegratedSpec(
                        n_bars=n_bars,
                        beta=1.0,
                        half_life=true_half_life,
                        sigma_spread=0.03,
                        sigma_common=0.01,
                    ),
                    seed=seed,
                )["true_spread"].to_numpy()
            )
            for seed in range(6)
        ]
        errors.append(abs(float(np.mean(estimates)) - true_half_life) / true_half_life)

    assert errors[1] < errors[0], (
        f"error did not shrink with sample size ({errors[0]:.4f} -> {errors[1]:.4f}); "
        "this would indicate bias rather than sampling noise"
    )
    assert errors[1] < 0.05, f"large-sample error {errors[1]:.4f} is too big"


def test_generated_spread_is_stationary():
    """The whole point of a cointegrated fixture: the spread must be I(0).

    ADF null hypothesis is a unit root; a small p-value rejects it, meaning
    stationary. If this failed, the "cointegrated" dataset would not be
    cointegrated and the strategy would have nothing real to trade.
    """
    spec = CointegratedSpec(
        n_bars=2000, beta=1.2, half_life=15.0, sigma_spread=0.03, sigma_common=0.012
    )
    df = generate_cointegrated(spec, seed=17)
    pvalue = adfuller(df["true_spread"].to_numpy(), autolag="AIC")[1]
    assert pvalue < 0.01, f"generated spread is not stationary (ADF p={pvalue})"


def test_individual_legs_are_non_stationary():
    """Each leg must be I(1) — otherwise the fixture is not modelling prices.

    A pair of stationary series would be trivially tradeable and would not
    exercise the cointegration machinery at all.
    """
    spec = CointegratedSpec(
        n_bars=2000, beta=1.2, half_life=15.0, sigma_spread=0.03, sigma_common=0.02
    )
    df = generate_cointegrated(spec, seed=19)
    for column in ("price_a", "price_b"):
        pvalue = adfuller(np.log(df[column].to_numpy()), autolag="AIC")[1]
        assert pvalue > 0.05, f"log {column} is unexpectedly stationary (p={pvalue})"


def test_independent_walks_have_a_non_stationary_spread():
    """The negative control must genuinely lack a stationary combination."""
    df = generate_independent(2000, seed=23)
    log_a = np.log(df["price_a"].to_numpy())
    log_b = np.log(df["price_b"].to_numpy())
    _, beta = ols_hedge_ratio(log_a, log_b)
    spread = log_a - beta * log_b
    pvalue = adfuller(spread, autolag="AIC")[1]
    assert pvalue > 0.05, (
        f"independent walks produced a stationary spread (p={pvalue}); "
        "the negative control is not controlling for anything"
    )
