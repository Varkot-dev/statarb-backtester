"""Tests for the significance layer.

The hand-rolled t-distribution is validated against ``scipy`` when available
and against published critical values otherwise, because a p-value computed
from a subtly wrong special function would be worse than no p-value at all —
it would look authoritative while being wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.significance import (
    _student_t_sf,
    assess,
    stationary_bootstrap_sharpe,
)


def _trades(pnl: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"pnl_net": pnl})


def _navs_from_returns(returns: np.ndarray, start: float = 100_000.0) -> np.ndarray:
    return start * np.cumprod(np.concatenate([[1.0], 1.0 + returns]))


# --------------------------------------------------------------------------
# The t-distribution
# --------------------------------------------------------------------------


def test_t_distribution_matches_published_critical_values():
    """Two-sided p at the classic 5% critical values must be ~0.05.

    These are the values in every statistics table, so they pin the
    implementation independently of any library.
    """
    for t_crit, df in ((2.228, 10), (2.086, 20), (2.009, 50), (1.984, 100)):
        assert _student_t_sf(t_crit, df) == pytest.approx(0.05, abs=1e-3)


def test_t_distribution_matches_scipy():
    """Exact agreement with scipy across the range the pipeline uses."""
    scipy_stats = pytest.importorskip("scipy.stats")
    for t in (0.0, 0.5, 0.861, 1.0, 1.96, 2.788, 5.0, 10.0):
        for df in (5, 10, 52, 53, 100, 500):
            assert _student_t_sf(t, df) == pytest.approx(
                2 * scipy_stats.t.sf(abs(t), df), abs=1e-10
            )


def test_t_distribution_edge_cases():
    """t=0 means no evidence at all, so p=1. The function is symmetric."""
    assert _student_t_sf(0.0, 10) == pytest.approx(1.0)
    assert _student_t_sf(2.5, 30) == pytest.approx(_student_t_sf(-2.5, 30))
    assert _student_t_sf(1e6, 10) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(_student_t_sf(1.0, 0))


# --------------------------------------------------------------------------
# The core claim: a no-edge result must not look significant
# --------------------------------------------------------------------------


def test_pure_noise_is_not_significant():
    """**The property this module exists for.**

    Trade PnL drawn from a zero-mean distribution must not be flagged as an
    edge. Averaged over many independent draws, the false-positive rate should
    be close to the nominal 5% — if it were much higher, the test would be
    rubber-stamping noise as skill.
    """
    rng = np.random.default_rng(42)
    n_experiments = 200
    false_positives = 0
    for _ in range(n_experiments):
        pnl = rng.normal(0.0, 1500.0, size=55)
        navs = _navs_from_returns(rng.normal(0.0, 0.001, size=300))
        result = assess(_trades(list(pnl)), navs, n_bootstrap=50)
        if result.is_significant:
            false_positives += 1

    rate = false_positives / n_experiments
    # Nominal 5%; allow up to 12% for binomial noise at n=200
    # (sd of the estimate is ~1.5 percentage points).
    assert rate < 0.12, (
        f"false-positive rate {rate:.1%} is far above the nominal 5%; "
        "the significance test is rubber-stamping noise"
    )


def test_a_genuine_edge_is_detected():
    """The complement to the noise test: a real edge must usually be found.

    Without this, a function that always returned "not significant" would pass
    the false-positive test above and be useless.

    Stated as *statistical power* across many draws rather than as a single
    seed. That distinction matters and is easy to get wrong: with a true mean
    of 400 and sd 1000 over 60 trades, the sample mean has a standard error of
    129, so individual draws land anywhere from roughly 140 to 660 and the
    t-statistic ranges from about 1.1 to 5. Asserting on one seed would be
    testing which seed was chosen.

    The claim here is that the test detects this effect size the large
    majority of the time. Theoretical power at a true t of 3.1 is around 87%;
    requiring 70% leaves room for sampling variation at 150 experiments while
    still failing loudly if the detector were broken.
    """
    rng = np.random.default_rng(7)
    n_experiments = 150
    detections = 0
    for _ in range(n_experiments):
        pnl = rng.normal(400.0, 1000.0, size=60)
        navs = _navs_from_returns(rng.normal(0.0004, 0.002, size=300))
        if assess(_trades(list(pnl)), navs, n_bootstrap=20).is_significant:
            detections += 1

    power = detections / n_experiments
    assert power > 0.70, (
        f"detected a real edge in only {power:.1%} of draws; "
        "the significance test lacks power"
    )


def test_an_unmistakable_edge_is_always_detected():
    """A deterministic case with no sampling ambiguity.

    PnL of exactly [1000] * 40 with a tiny amount of dispersion gives an
    enormous t-statistic, so this must be significant regardless of seed. This
    is the sanity floor beneath the power test above.
    """
    pnl = [1000.0 + (10.0 if i % 2 else -10.0) for i in range(40)]
    result = assess(_trades(pnl), np.array([]))
    assert result.is_significant
    assert result.t_statistic > 100.0
    assert result.p_value < 1e-9


def test_t_statistic_is_hand_computable():
    """Pin the arithmetic against a fixture computed by hand.

    PnL [100, 200, 300, 400, 500]: mean 300, sample sd = sqrt(25000) =
    158.113883, SE = 158.113883 / sqrt(5) = 70.710678, t = 300 / 70.710678
    = 4.2426407.
    """
    result = assess(_trades([100.0, 200.0, 300.0, 400.0, 500.0]), np.array([]))
    assert result.mean_trade_pnl == pytest.approx(300.0)
    assert result.se_trade_pnl == pytest.approx(70.710678, abs=1e-5)
    assert result.t_statistic == pytest.approx(4.2426407, abs=1e-6)


def test_losing_strategy_has_a_negative_t_statistic():
    """No absolute value anywhere: a losing strategy reports a negative t."""
    rng = np.random.default_rng(11)
    pnl = rng.normal(-350.0, 900.0, size=60)
    result = assess(_trades(list(pnl)), np.array([]))
    assert result.t_statistic < 0
    assert result.mean_trade_pnl < 0


def test_too_few_trades_is_reported_not_guessed():
    """With 0 or 1 trades there is no dispersion to test against."""
    for pnl in ([], [500.0]):
        result = assess(_trades(pnl), np.array([]))
        assert not result.is_significant
        assert "too few trades" in result.verdict()


def test_zero_variance_pnl_does_not_divide_by_zero():
    """Identical trade PnLs give zero SE; must not produce inf or nan."""
    result = assess(_trades([100.0] * 10), np.array([]))
    assert np.isfinite(result.t_statistic)


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def test_bootstrap_is_reproducible():
    """Same seed, identical resamples — the reported CI must be stable."""
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0003, 0.004, size=400)
    a = stationary_bootstrap_sharpe(returns, 252.0, 0.04, n_bootstrap=100, seed=99)
    b = stationary_bootstrap_sharpe(returns, 252.0, 0.04, n_bootstrap=100, seed=99)
    np.testing.assert_array_equal(a, b)


def test_bootstrap_centres_on_the_sample_sharpe():
    """The bootstrap distribution should be centred near the point estimate."""
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0005, 0.003, size=800)
    rf_bar = 1.04 ** (1 / 252) - 1
    excess = returns - rf_bar
    point = excess.mean() / excess.std(ddof=1) * np.sqrt(252)

    boot = stationary_bootstrap_sharpe(returns, 252.0, 0.04, n_bootstrap=400, seed=17)
    assert float(np.median(boot)) == pytest.approx(point, abs=0.6)


def test_bootstrap_interval_brackets_zero_for_a_no_edge_series():
    """A zero-mean return series must yield a CI that contains zero."""
    rng = np.random.default_rng(23)
    rf_bar = 1.04 ** (1 / 252) - 1
    # Returns centred exactly on the risk-free rate: zero excess by construction.
    returns = rng.normal(rf_bar, 0.004, size=1000)
    navs = _navs_from_returns(returns)
    result = assess(_trades(list(rng.normal(0, 1000, 50))), navs, n_bootstrap=400)
    assert result.sharpe_ci_low < 0 < result.sharpe_ci_high, (
        f"CI [{result.sharpe_ci_low:.3f}, {result.sharpe_ci_high:.3f}] "
        "should contain zero for a no-edge series"
    )


def test_bootstrap_handles_a_short_series():
    """Fewer than two observations yields an empty array, not a crash."""
    assert len(stationary_bootstrap_sharpe(np.array([0.01]), 252.0, 0.04)) == 0
    assert len(stationary_bootstrap_sharpe(np.array([]), 252.0, 0.04)) == 0


def test_sharpe_se_has_correct_nominal_coverage():
    """**The test that was missing.** A confidence interval must cover.

    The previous test asserted only that the SE shrinks with sample size — a
    property that holds for any multiple of the correct value, and which passed
    while the SE was understated by sqrt(252). A nominal 95% interval achieved
    measured coverage of 0.095.

    Coverage is the property that actually matters, so coverage is what is
    asserted. This is the same unit-mixing error CREDITS.md criticises another
    project for; catching it there and shipping it here is why the check is now
    on the outcome rather than on the formula.
    """
    bars_per_year, n_bars = 252.0, 2515
    for true_annual_sharpe in (0.0, 0.65):
        per_bar_mean = true_annual_sharpe / np.sqrt(bars_per_year) * 0.01
        covered = 0
        n_experiments = 400
        for seed in range(n_experiments):
            returns = np.random.default_rng(seed).normal(per_bar_mean, 0.01, n_bars)
            navs = 100_000.0 * np.cumprod(np.concatenate([[1.0], 1.0 + returns]))
            result = assess(
                _trades([1.0, 2.0]), navs, risk_free_annual=0.0, n_bootstrap=2
            )
            if abs(result.sharpe - true_annual_sharpe) <= 1.96 * result.sharpe_se:
                covered += 1
        coverage = covered / n_experiments
        assert 0.90 <= coverage <= 0.99, (
            f"nominal 95% interval achieved {coverage:.3f} coverage at "
            f"true Sharpe {true_annual_sharpe}; check for a units mismatch "
            f"between the annualized Sharpe and the per-bar sample size"
        )


def test_sharpe_se_and_bootstrap_ci_are_the_same_order_of_magnitude():
    """Two uncertainty measures on one series must not disagree wildly.

    They will not agree exactly — the bootstrap accounts for serial dependence
    the IID formula ignores, so it is legitimately wider. But a factor of 25
    between them, which is what the units bug produced, means one of them is
    simply wrong. This is the cheap cross-check that would have caught it from
    the shipped output alone.
    """
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0004, 0.004, 2515)
    navs = 100_000.0 * np.cumprod(np.concatenate([[1.0], 1.0 + returns]))
    result = assess(_trades(list(rng.normal(200, 800, 50))), navs, n_bootstrap=400)

    bootstrap_half_width = (result.sharpe_ci_high - result.sharpe_ci_low) / 2.0
    ratio = bootstrap_half_width / result.sharpe_se
    assert 0.3 < ratio < 4.0, (
        f"bootstrap half-width {bootstrap_half_width:.4f} and Lo SE "
        f"{result.sharpe_se:.4f} differ by {ratio:.1f}x; one of them is wrong"
    )


def test_sharpe_standard_error_grows_as_the_sample_shrinks():
    """Lo (2002): SE ~ sqrt((1 + SR^2/2)/n), so it must fall with n."""
    rng = np.random.default_rng(29)
    long_navs = _navs_from_returns(rng.normal(0.0004, 0.003, size=2000))
    short_navs = _navs_from_returns(rng.normal(0.0004, 0.003, size=200))
    trades = _trades(list(rng.normal(200, 800, 40)))
    long_result = assess(trades, long_navs, n_bootstrap=50)
    short_result = assess(trades, short_navs, n_bootstrap=50)
    assert long_result.sharpe_se < short_result.sharpe_se


def test_verdict_is_readable():
    """The one-line summary must state the direction plainly."""
    rng = np.random.default_rng(31)
    significant = assess(
        _trades(list(rng.normal(500, 800, 60))), np.array([]), n_bootstrap=20
    )
    assert "distinguishable from zero" in significant.verdict()
    assert "NOT" not in significant.verdict()

    noise = assess(_trades(list(rng.normal(0, 1500, 50))), np.array([]), n_bootstrap=20)
    if not noise.is_significant:
        assert "NOT distinguishable" in noise.verdict()
        assert "consistent with no edge" in noise.verdict()


def test_result_serialises_completely():
    """Every field must reach summary.json."""
    result = assess(_trades([100.0, -50.0, 200.0]), np.array([]))
    expected = {
        "n_trades", "mean_trade_pnl", "se_trade_pnl", "t_statistic", "p_value",
        "sharpe", "sharpe_se", "sharpe_ci_low", "sharpe_ci_high",
        "bootstrap_p_value", "is_significant",
    }
    assert set(result.to_dict().keys()) == expected
