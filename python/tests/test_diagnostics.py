"""Tests for the diagnostics layer.

The window/half-life finding is the substantive claim this repository makes, so
these tests pin it: the relationship must hold, and the specification-search
correction must actually reject a searched-for result.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from statarb.diagnostics import (
    MIN_WINDOW_HALF_LIFE_RATIO,
    deflated_sharpe_threshold,
    expected_max_sharpe_under_null,
    half_life_sensitivity,
    sharpe_standard_error,
    walk_forward,
    window_ratio_sensitivity,
)
from statarb.synth import CointegratedSpec, generate_cointegrated

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENGINE = REPO_ROOT / "engine" / "_build" / "default" / "bin" / "main.exe"

requires_engine = pytest.mark.skipif(
    not ENGINE.exists(), reason="engine not built; run 'make build'"
)


# --------------------------------------------------------------------------
# Expected maximum under the null
# --------------------------------------------------------------------------


def test_expected_max_grows_with_trials():
    """More trials means a higher best-from-luck. The whole point."""
    values = [expected_max_sharpe_under_null(n, 2515) for n in (2, 5, 30, 100, 1000)]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_expected_max_is_zero_for_a_single_trial():
    """With one trial there is no selection effect to correct for."""
    assert expected_max_sharpe_under_null(1, 2515) == 0.0
    assert expected_max_sharpe_under_null(0, 2515) == 0.0


def test_expected_max_shrinks_with_sample_length():
    """A longer sample estimates Sharpe more precisely, so luck buys less."""
    short = expected_max_sharpe_under_null(30, 250)
    long = expected_max_sharpe_under_null(30, 5000)
    assert long < short


def test_expected_max_agrees_with_the_gumbel_approximation():
    """Cross-check the two-term form against the asymptotic Gumbel expansion.

    Two independent approximations to E[max of N standard normals] should agree
    closely. They are derived differently, so agreement is real evidence rather
    than a tautology.

    The Gumbel form is the one used by ``mnemox-ai/deflated-sharpe``
    (Apache-2.0); reading that implementation is what prompted this check.
    """
    n_obs = 2515
    for n_trials in (5, 30, 100, 1000):
        mine = expected_max_sharpe_under_null(n_trials, n_obs, bars_per_year=1.0)
        ln_m = math.log(n_trials)
        z = math.sqrt(2 * ln_m)
        e_z = z - (math.log(math.log(n_trials)) + math.log(4 * math.pi)) / (2 * z)
        gumbel = e_z * math.sqrt(1.0 / (n_obs - 1))
        assert abs(mine - gumbel) < 0.01, (
            f"N={n_trials}: two-term {mine:.5f} vs Gumbel {gumbel:.5f}"
        )


# --------------------------------------------------------------------------
# Sharpe standard error
# --------------------------------------------------------------------------


def test_standard_error_reduces_to_gaussian_at_zero_sharpe():
    """At SR = 0, skew 0 and kurtosis 3, the correction must vanish exactly.

    Note it reduces exactly only at SR = 0. The (kurtosis-1)/4 * SR^2 term is
    non-zero for any non-zero Sharpe even under normality — with kurtosis 3 that
    is +0.5*SR^2, so a normal-returns Sharpe of 0.05 already carries a ~0.06%
    wider standard error than the naive sqrt(1/T). Small, but the formula is not
    the Gaussian one in general and pretending otherwise would misstate it.
    """
    n = 1000
    assert sharpe_standard_error(0.0, n, 0.0, 3.0) == pytest.approx(
        math.sqrt(1.0 / (n - 1)), rel=1e-12
    )
    # At a small but non-zero Sharpe the correction is present and positive.
    assert sharpe_standard_error(0.05, n, 0.0, 3.0) > math.sqrt(1.0 / (n - 1))


def test_fat_tails_widen_the_standard_error():
    """Excess kurtosis must make the estimate LESS certain, not more.

    Getting this backwards would understate uncertainty on exactly the
    fat-tailed return distribution a pairs strategy produces — the direction
    that flatters the result.
    """
    normal = sharpe_standard_error(0.30, 1000, 0.0, 3.0)
    fat = sharpe_standard_error(0.30, 1000, 0.0, 14.0)
    assert fat > normal


def test_negative_skew_widens_the_standard_error():
    """Left-skewed returns (small wins, big losses) carry more uncertainty."""
    symmetric = sharpe_standard_error(0.30, 1000, 0.0, 3.0)
    left_skewed = sharpe_standard_error(0.30, 1000, -1.5, 3.0)
    assert left_skewed > symmetric


# --------------------------------------------------------------------------
# The deflation gate
# --------------------------------------------------------------------------


def test_a_searched_result_at_the_luck_threshold_does_not_survive():
    """**The core guard.**

    The measured case: best Sharpe +0.654 after ~30 configurations on 2515 bars.
    The expected best from pure luck at that search size is ~+0.656. The gate
    must reject it.
    """
    result = deflated_sharpe_threshold(0.654, n_trials=30, n_observations=2515)
    assert not result["survives"]
    assert result["p_value"] > 0.4, "a coin-flip result should have p near 0.5"
    assert abs(result["deflated_sharpe_statistic"]) < 0.5


def test_a_genuinely_strong_result_survives():
    """The complement: a large Sharpe from few trials must pass."""
    result = deflated_sharpe_threshold(2.5, n_trials=3, n_observations=2515)
    assert result["survives"]
    assert result["p_value"] < 0.01


def test_more_trials_makes_the_same_sharpe_less_credible():
    """Monotonicity in the search size — the property the gate exists for."""
    p_values = [
        deflated_sharpe_threshold(0.8, n_trials=n, n_observations=2515)["p_value"]
        for n in (2, 10, 50, 500)
    ]
    assert all(a < b for a, b in zip(p_values, p_values[1:])), (
        f"p-value must rise with trials, got {p_values}"
    )


def test_units_are_handled_consistently():
    """Guard against the annualized/per-bar mix-up.

    An annualized Sharpe of 0.654 on 2515 bars is an unremarkable result. If the
    statistic came back in the tens, the implementation would be comparing an
    annualized numerator against a per-bar denominator — a factor of
    sqrt(252) ≈ 15.9. That is a real bug observed in a public implementation of
    this same formula, so it is pinned here.
    """
    result = deflated_sharpe_threshold(0.654, n_trials=30, n_observations=2515)
    assert abs(result["deflated_sharpe_statistic"]) < 5.0, (
        "statistic is implausibly large; check for an annualization mismatch"
    )


# --------------------------------------------------------------------------
# The window/half-life finding
# --------------------------------------------------------------------------


@requires_engine
def test_short_windows_break_the_signal(tmp_path):
    """**The finding.** Same data, same strategy — only the window changes.

    Sharpe at a window/half-life ratio below the threshold must be materially
    worse than well above it. This is the claim the README makes, so it is
    pinned rather than left to a chart.
    """
    rows = window_ratio_sensitivity(
        ENGINE, tmp_path, half_life=20.0, ratios=(2.0, 3.0, 6.0, 12.5)
    )
    assert len(rows) == 4
    by_ratio = {r.window_ratio: r for r in rows}

    too_short = max(by_ratio[2.0].sharpe, by_ratio[3.0].sharpe)
    long_enough = min(by_ratio[6.0].sharpe, by_ratio[12.5].sharpe)
    assert long_enough > too_short + 0.3, (
        f"a sufficient window should clearly beat an insufficient one: "
        f"{long_enough:.3f} vs {too_short:.3f}"
    )
    # And the win rate should climb with the ratio, which is the mechanism.
    assert by_ratio[12.5].win_rate > by_ratio[2.0].win_rate


@requires_engine
def test_slower_reversion_degrades_performance(tmp_path):
    """At a fixed window, a longer half-life must hurt.

    This is the same finding viewed from the other axis: holding the window at
    60 while the half-life rises drives the ratio down through the threshold.
    """
    rows = half_life_sensitivity(
        ENGINE, tmp_path, half_lives=(5.0, 30.0), zscore_window=60
    )
    assert len(rows) == 2
    fast, slow = rows[0], rows[1]
    assert fast.sharpe > slow.sharpe + 0.5
    # The mechanism check: it is not simply that there are fewer trades.
    assert slow.n_trades > fast.n_trades * 0.5, (
        "trade count should not collapse; the effect is worse trades, not fewer"
    )


@requires_engine
def test_walk_forward_is_out_of_sample(tmp_path):
    """The window must be selected without touching the evaluation segment."""
    spec = CointegratedSpec(
        n_bars=3000, beta=1.1, half_life=12.0, sigma_spread=0.04, sigma_common=0.008
    )
    prices = generate_cointegrated(spec, seed=99)
    row = walk_forward(ENGINE, tmp_path, prices, "SYNTH")
    assert row is not None
    assert row.selected_window == pytest.approx(
        round(row.in_sample_half_life * MIN_WINDOW_HALF_LIFE_RATIO), abs=1
    )
    assert np.isfinite(row.oos_sharpe)


@requires_engine
def test_walk_forward_declines_when_the_sample_is_too_short(tmp_path):
    """A long half-life needs a long window, and eventually there is no test left.

    Returning None is the honest outcome — better than silently capping the
    window and reporting a result computed under a ratio the method says is
    insufficient.
    """
    spec = CointegratedSpec(
        n_bars=400, beta=1.0, half_life=60.0, sigma_spread=0.03, sigma_common=0.01
    )
    prices = generate_cointegrated(spec, seed=5)
    assert walk_forward(ENGINE, tmp_path, prices, "SHORT") is None


# --------------------------------------------------------------------------
# Walk-forward selection
# --------------------------------------------------------------------------


def test_walk_forward_summary_refuses_to_over_read_a_small_sample():
    """**The guard against reporting a 2-pair average as a result.**

    The drop rule discards any pair whose half-life demands a window longer
    than the out-of-sample segment supports. Measured on no-edge data it keeps
    only ~3.6%, so the surviving mean carries a standard error around 0.23 —
    wide enough that a 2-pair average cannot distinguish an edge from noise.

    The summary must say so rather than reporting the mean bare.
    """
    from statarb.diagnostics import WalkForwardRow, WalkForwardSummary

    rows = [
        WalkForwardRow("A/B", 30.0, 150, 0.44, 7, 1.03),
        WalkForwardRow("C/D", 43.0, 214, -0.13, 5, -0.27),
    ]
    summary = WalkForwardSummary(rows=rows, n_dropped=3, dropped_pairs=["E/F", "G/H", "I/J"])

    assert summary.n_kept == 2
    assert summary.mean_oos_sharpe == pytest.approx(0.155, abs=0.01)
    assert np.isfinite(summary.se_mean_oos_sharpe)
    verdict = summary.verdict()
    assert "only 2 of 5" in verdict
    assert "cannot distinguish an edge from noise" in verdict


def test_walk_forward_summary_reports_the_standard_error_when_it_can():
    """With enough survivors the mean is reported with its uncertainty."""
    from statarb.diagnostics import WalkForwardRow, WalkForwardSummary

    rows = [
        WalkForwardRow(f"P{i}", 30.0, 150, 0.1 * i, 10, 0.5) for i in range(5)
    ]
    summary = WalkForwardSummary(rows=rows, n_dropped=1, dropped_pairs=["Q"])
    assert "±" in summary.verdict() or "SE" in summary.verdict()
    assert summary.se_mean_oos_sharpe > 0


def test_walk_forward_summary_handles_zero_survivors():
    """No survivor means no conclusion, stated plainly rather than as nan."""
    from statarb.diagnostics import WalkForwardSummary

    summary = WalkForwardSummary(rows=[], n_dropped=5, dropped_pairs=list("ABCDE"))
    assert summary.n_kept == 0
    assert np.isnan(summary.mean_oos_sharpe)
    assert "no out-of-sample conclusion" in summary.verdict()
