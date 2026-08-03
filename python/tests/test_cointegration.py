"""Tests for the cointegration and pair-selection layer.

The most important test here is
:func:`test_correlation_alone_is_insufficient`, which demonstrates
empirically the claim the README makes: two independent random walks can be
highly correlated in levels while having no cointegrating relationship. That is
the fact that motivates the whole selection procedure, so it is verified rather
than merely asserted in prose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.cointegration import (
    analyse_pair,
    correlation_matrix,
    estimate_half_life,
    ols_hedge_ratio,
    screen_universe,
)
from statarb.synth import CointegratedSpec, generate_cointegrated, generate_independent


def test_ols_hedge_ratio_on_an_exact_line():
    """An exact linear relationship recovers its coefficients exactly."""
    x = np.linspace(1.0, 10.0, 50)
    y = 3.0 + 2.5 * x
    alpha, beta = ols_hedge_ratio(y, x)
    assert alpha == pytest.approx(3.0, abs=1e-9)
    assert beta == pytest.approx(2.5, abs=1e-9)


def test_half_life_of_a_known_ar1():
    """A constructed AR(1) with known phi must give back its half-life."""
    rng = np.random.default_rng(5)
    half_life = 20.0
    phi = np.exp(-np.log(2.0) / half_life)
    n = 8000
    series = np.empty(n)
    series[0] = 0.0
    innovations = rng.normal(0.0, 0.05, size=n)
    for t in range(1, n):
        series[t] = phi * series[t - 1] + innovations[t]
    assert estimate_half_life(series) == pytest.approx(half_life, rel=0.15)


def test_half_life_of_a_random_walk_is_untradeable():
    """A random walk yields either no half-life or one far too long to trade.

    It is tempting to assert ``== inf`` here, and that assertion fails — for a
    reason worth stating, because it is the same reason the Dickey-Fuller test
    needs its own critical values.

    The OU regression estimates :math:`\\lambda` in
    :math:`\\Delta s_t = \\alpha + \\lambda s_{t-1} + \\varepsilon_t`. For a true
    random walk the population value is :math:`\\lambda = 0`, but the OLS
    estimator is *biased downward*: :math:`s_{t-1}` is correlated with the
    innovation that produced it, so :math:`\\hat\\lambda` has a negative mean in
    finite samples. Empirically, across seeds, roughly seven draws in eight
    give :math:`\\hat\\lambda < 0` and hence a large but finite half-life
    (hundreds to thousands of bars); the rest give ``inf``.

    That downward bias is precisely why a naive OLS t-test on
    :math:`\\hat\\lambda` over-rejects the unit-root null, and why ADF exists.

    So the property that actually matters — and the one the strategy depends
    on — is not that the number is infinite but that it is **outside the
    tradeable band**. A half-life of 350 bars on daily data is 1.4 years; no
    position is held that long, and :meth:`PairStats.is_tradeable` rejects it.
    """
    max_tradeable_half_life = 120.0
    untradeable = 0
    n_seeds = 12
    for seed in range(n_seeds):
        walk = np.cumsum(np.random.default_rng(seed).normal(0.0, 1.0, size=5000))
        half_life = estimate_half_life(walk)
        # Either no mean reversion at all, or reversion far too slow to trade.
        assert half_life > max_tradeable_half_life, (
            f"seed {seed}: a random walk produced a tradeable half-life of "
            f"{half_life:.1f} bars, which would let the strategy trade noise"
        )
        untradeable += 1
    assert untradeable == n_seeds


def test_half_life_of_a_degenerate_series():
    """Too few points, or a constant series, yields infinity rather than raising."""
    assert estimate_half_life(np.array([1.0, 2.0])) == float("inf")
    assert estimate_half_life(np.zeros(100)) == float("inf")


def test_cointegrated_pair_is_detected():
    """A pair built to be cointegrated must pass both tests."""
    spec = CointegratedSpec(
        n_bars=1500, beta=1.2, half_life=15.0, sigma_spread=0.03, sigma_common=0.012
    )
    df = generate_cointegrated(spec, seed=31)
    stats = analyse_pair(df["price_a"], df["price_b"])

    assert stats.eg_pvalue < 0.05, f"Engle-Granger failed to detect it (p={stats.eg_pvalue})"
    assert stats.adf_pvalue < 0.05, f"ADF on the residual failed (p={stats.adf_pvalue})"
    assert stats.johansen_trace_stat > stats.johansen_crit_95, "Johansen failed to detect it"
    assert stats.is_cointegrated()
    assert stats.is_tradeable()
    assert stats.hedge_ratio == pytest.approx(spec.beta, rel=0.10)


def test_independent_walks_are_not_cointegrated():
    """The negative control: no relationship, so no cointegration."""
    df = generate_independent(1500, seed=37)
    stats = analyse_pair(df["price_a"], df["price_b"])
    assert not stats.is_cointegrated(), (
        f"independent walks reported as cointegrated "
        f"(EG p={stats.eg_pvalue}, Johansen {stats.johansen_trace_stat} "
        f"vs {stats.johansen_crit_95})"
    )


def test_correlation_alone_is_insufficient():
    """**The core claim.** High level-correlation without cointegration.

    Two independent random walks are generated repeatedly; the test asserts
    that at least one draw shows high correlation of *log levels* while
    failing the cointegration tests. This is the Granger-Newbold spurious
    regression, and it is exactly why a correlation screen selects the wrong
    pairs.

    Several seeds are tried because whether a given pair of walks happens to
    drift together is itself random; the claim is that it happens *often*, not
    that it happens for every seed.
    """
    found = None
    for seed in range(60):
        df = generate_independent(600, seed=1000 + seed)
        log_a = np.log(df["price_a"].to_numpy())
        log_b = np.log(df["price_b"].to_numpy())
        level_correlation = abs(np.corrcoef(log_a, log_b)[0, 1])
        if level_correlation < 0.85:
            continue
        stats = analyse_pair(df["price_a"], df["price_b"])
        if not stats.is_cointegrated():
            found = (level_correlation, stats)
            break

    assert found is not None, (
        "expected to find at least one highly level-correlated but "
        "non-cointegrated pair among 60 draws of independent random walks"
    )
    level_correlation, stats = found
    assert level_correlation > 0.85
    assert not stats.is_cointegrated()
    # And the honest measure — return correlation — is near zero, which is what
    # a returns-based screen would correctly report.
    assert abs(stats.correlation_returns) < 0.20


def test_correlation_matrix_defaults_to_returns():
    """The default must be return correlation, not level correlation.

    Level correlations between I(1) series are spurious, so defaulting to them
    would build the wrong screen into the library.
    """
    rng = np.random.default_rng(41)
    n = 500
    # Two independent walks that both drift upward.
    prices = pd.DataFrame(
        {
            "X": np.exp(np.log(100) + np.cumsum(rng.normal(0.001, 0.01, n))),
            "Y": np.exp(np.log(50) + np.cumsum(rng.normal(0.001, 0.01, n))),
        }
    )
    returns_corr = correlation_matrix(prices, use_returns=True)
    levels_corr = correlation_matrix(prices, use_returns=False)

    assert abs(returns_corr.loc["X", "Y"]) < 0.20, "independent returns should be uncorrelated"
    # The level correlation is typically far higher — the spurious effect.
    assert abs(levels_corr.loc["X", "Y"]) > abs(returns_corr.loc["X", "Y"])
    # Diagonals are 1 by definition.
    assert returns_corr.loc["X", "X"] == pytest.approx(1.0)


def test_analyse_pair_rejects_bad_input():
    """Validation is loud, so a data problem never becomes a silent sample change."""
    good = pd.Series(np.linspace(100, 110, 100))

    with pytest.raises(ValueError, match="lengths differ"):
        analyse_pair(good, pd.Series(np.linspace(50, 55, 99)))

    with pytest.raises(ValueError, match="at least 30"):
        analyse_pair(good.iloc[:10], pd.Series(np.linspace(50, 55, 10)))

    with pytest.raises(ValueError, match="strictly positive"):
        bad = good.copy()
        bad.iloc[5] = -1.0
        analyse_pair(bad, pd.Series(np.linspace(50, 55, 100)))

    with pytest.raises(ValueError, match="finite"):
        bad = good.copy()
        bad.iloc[5] = np.inf
        analyse_pair(bad, pd.Series(np.linspace(50, 55, 100)))


def test_is_tradeable_filters_on_half_life():
    """Cointegration alone is not enough; the timescale must be usable."""
    spec = CointegratedSpec(
        n_bars=1500, beta=1.0, half_life=15.0, sigma_spread=0.03, sigma_common=0.012
    )
    df = generate_cointegrated(spec, seed=43)
    stats = analyse_pair(df["price_a"], df["price_b"])
    assert stats.is_cointegrated()
    # A cointegrated pair whose half-life falls outside the usable band is
    # correctly rejected for trading even though the statistics pass.
    assert not stats.is_tradeable(min_half_life=200.0, max_half_life=400.0)
    assert stats.is_tradeable(min_half_life=1.0, max_half_life=120.0)


def test_screen_universe_ranks_by_evidence():
    """A universe containing one genuine pair must surface it."""
    spec = CointegratedSpec(
        n_bars=900, beta=1.1, half_life=15.0, sigma_spread=0.03, sigma_common=0.02
    )
    coint = generate_cointegrated(spec, seed=47)
    noise = generate_independent(900, seed=53)

    prices = pd.DataFrame(
        {
            "COINT_A": coint["price_a"].to_numpy(),
            "COINT_B": coint["price_b"].to_numpy(),
            "NOISE_A": noise["price_a"].to_numpy(),
            "NOISE_B": noise["price_b"].to_numpy(),
        }
    )
    # A permissive pre-filter so every pair reaches the cointegration stage;
    # the point of this test is the ranking, not the filter.
    result = screen_universe(prices, min_abs_correlation=0.0)

    assert len(result) == 6, "expected all 4-choose-2 pairs"
    tradeable = result[result["is_tradeable"]]
    assert len(tradeable) >= 1, "the genuine pair should be found"
    pairs = {frozenset((row.name_a, row.name_b)) for row in tradeable.itertuples()}
    assert frozenset(("COINT_A", "COINT_B")) in pairs


def test_screen_universe_handles_an_empty_result():
    """A screen that finds nothing returns an empty frame with the right columns."""
    noise = generate_independent(400, seed=59)
    prices = pd.DataFrame(
        {"A": noise["price_a"].to_numpy(), "B": noise["price_b"].to_numpy()}
    )
    # A pre-filter nothing can pass.
    result = screen_universe(prices, min_abs_correlation=0.999)
    assert len(result) == 0
    assert "eg_pvalue" in result.columns


def test_pair_stats_serialises_completely():
    """Every field must reach the CSV, or a diagnostic silently disappears."""
    spec = CointegratedSpec(
        n_bars=400, beta=1.0, half_life=12.0, sigma_spread=0.03, sigma_common=0.012
    )
    df = generate_cointegrated(spec, seed=61)
    row = analyse_pair(df["price_a"], df["price_b"]).to_dict()
    expected = {
        "name_a", "name_b", "correlation_levels", "correlation_returns",
        "hedge_ratio", "eg_pvalue", "adf_pvalue", "johansen_trace_stat",
        "johansen_crit_95", "half_life", "spread_std", "n_obs",
        "is_cointegrated", "is_tradeable",
    }
    assert set(row.keys()) == expected
