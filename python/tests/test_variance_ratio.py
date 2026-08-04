"""Tests for the Lo-MacKinlay variance ratio test.

The claim this module makes is that it supplies a *calibrated* p-value on the
random-walk null — the thing the half-life estimator cannot supply. A test
suite that only checked "mean-reverting series give VR < 1" would not test that
claim at all, since an uncalibrated statistic can get the sign right and still
reject random walks half the time.

So the load-bearing tests here are the two that would catch a miscalibration:
the false-positive rate across many seeded random walks must land near the
nominal 5%, and the point estimate must match the closed form for an AR(1),
where the true variance ratio is known exactly rather than approximated.
"""

from __future__ import annotations

import numpy as np
import pytest

from statarb.synth import CointegratedSpec, generate_cointegrated, generate_independent
from statarb.variance_ratio import (
    DEFAULT_HORIZONS,
    MIN_OBSERVATIONS,
    interpret,
    variance_ratio,
    variance_ratio_test,
)


def random_walk(n: int, seed: int, sigma: float = 1.0) -> np.ndarray:
    """A pure random walk in levels — the null hypothesis made concrete."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0.0, sigma, size=n))


def ar1(n: int, phi: float, seed: int, sigma: float = 1.0) -> np.ndarray:
    """A stationary AR(1) in levels, started from its stationary distribution.

    Starting at the stationary variance rather than at zero matters for the
    closed-form comparison: a burn-in during which the series is anomalously
    close to its mean would bias the measured variance ratio downward.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sigma / np.sqrt(1.0 - phi**2))
    innovations = rng.normal(0.0, sigma, size=n)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + innovations[t]
    return out


def theoretical_ar1_variance_ratio(phi: float, q: int) -> float:
    """Closed-form variance ratio for a stationary AR(1) in levels.

    Derivation. For :math:`s_t = \\phi s_{t-1} + \\varepsilon_t` with
    stationary variance :math:`\\sigma_s^2`, the autocovariance is
    :math:`\\gamma_k = \\sigma_s^2 \\phi^k`. The variance of a q-bar change is

    .. math:: \\mathrm{Var}(s_t - s_{t-q}) = 2\\gamma_0 - 2\\gamma_q
              = 2\\sigma_s^2 (1 - \\phi^q)

    and the one-bar case is that with ``q = 1``. The variance ratio is the
    former over ``q`` times the latter, so the :math:`2\\sigma_s^2` cancels:

    .. math:: VR(q) = \\frac{1 - \\phi^q}{q\\,(1 - \\phi)}

    Two sanity checks on the form. As :math:`\\phi \\to 1` the series becomes a
    random walk and L'Hopital gives :math:`VR \\to 1`. At :math:`q = 1` it is
    identically 1, as it must be. And it is below 1 for every
    :math:`\\phi \\in (0, 1)` and :math:`q \\geq 2`, which is the statement
    that a stationary series is mean-reverting.
    """
    return (1.0 - phi**q) / (q * (1.0 - phi))


# --------------------------------------------------------------------------
# Calibration under the null — the test that matters most
# --------------------------------------------------------------------------


def test_a_random_walk_is_not_rejected_on_average():
    """The single case the half-life estimator gets wrong.

    On a genuine random walk ``estimate_half_life`` returns a large but finite
    number that invites a mean-reversion reading. The variance ratio test must
    return a ratio near 1 and, in aggregate, decline to reject.

    Averaged over seeds rather than asserted on one. A correctly calibrated
    test rejects 5% of genuine random walks *by design*, so demanding that
    some hand-picked seed clears every horizon would be selecting the seed to
    fit the assertion — and would silently break the moment anything upstream
    perturbed the PRNG stream. What must hold is that the mean ratio sits at
    1.0 and that rejections stay rare, which is what is checked here; the rate
    itself is pinned precisely by the calibration test below.
    """
    ratios: list[float] = []
    rejections = 0
    n_seeds = 40

    for seed in range(n_seeds):
        for result in variance_ratio_test(random_walk(2500, seed=seed)):
            ratios.append(result.variance_ratio)
            rejections += result.rejects_random_walk()

    assert float(np.mean(ratios)) == pytest.approx(1.0, abs=0.02), (
        f"mean VR over {n_seeds} random walks is {np.mean(ratios):.4f}, not 1.0; "
        f"the estimator is biased"
    )
    assert rejections / len(ratios) < 0.12, (
        f"rejected {rejections}/{len(ratios)} horizon-tests on genuine random "
        f"walks; too high even allowing for horizon dependence"
    )


def test_false_positive_rate_is_near_nominal():
    """**The calibration guard.**

    A test that rejects the random-walk null at 5% must do so on about 5% of
    actual random walks. If the rate came back at 20%, every "significant"
    reversion this module reports on real data would be suspect, and the sign
    checks below would all still pass — so this is the test that establishes
    the p-values mean anything.

    Both nulls are measured. The homoskedastic one is expected to be
    approximately calibrated *here* because these are Gaussian IID
    increments — the assumption it makes is true by construction. That is the
    point: it is calibrated on data that satisfies it, and the module docstring
    warns against trusting it on real spreads that do not.

    The band is wide because 200 trials is a small sample for a 5% rate: the
    binomial standard error is about 1.5 percentage points, so anything inside
    roughly 1% to 11% is consistent with nominal. A miscalibration bad enough
    to matter would land far outside it.
    """
    n_trials = 200
    rejections_robust = 0
    rejections_homoskedastic = 0

    for seed in range(n_trials):
        series = random_walk(1000, seed=10_000 + seed)
        # One horizon only. Counting "rejected at any of four horizons" would
        # measure a different and larger quantity, because the horizons are
        # positively dependent rather than independent.
        result = variance_ratio_test(series, horizons=(4,))[0]
        rejections_robust += result.p_robust < 0.05
        rejections_homoskedastic += result.p_homoskedastic < 0.05

    rate_robust = rejections_robust / n_trials
    rate_homoskedastic = rejections_homoskedastic / n_trials

    assert 0.01 <= rate_robust <= 0.11, (
        f"robust false-positive rate {rate_robust:.3f} is far from the nominal "
        f"0.05; the test is miscalibrated and its p-values cannot be trusted"
    )
    assert 0.01 <= rate_homoskedastic <= 0.11, (
        f"homoskedastic false-positive rate {rate_homoskedastic:.3f} is far "
        f"from nominal on Gaussian IID increments, which satisfy its own "
        f"assumption exactly"
    )


def test_independent_random_walks_are_not_rejected():
    """The repository's negative-control generator, run through this test.

    ``generate_independent`` is the dataset the README uses to show the
    strategy failing where it should. Each leg is a random walk, so neither
    log-price series should be distinguishable from one.
    """
    prices = generate_independent(n_bars=2000, seed=42)
    for column in ("price_a", "price_b"):
        log_prices = np.log(prices[column].to_numpy())
        for result in variance_ratio_test(log_prices):
            assert result.p_robust > 0.05, (
                f"{column} q={result.horizon}: rejected the random-walk null on "
                f"a leg that is a random walk by construction "
                f"(p={result.p_robust:.4f})"
            )


# --------------------------------------------------------------------------
# The closed form
# --------------------------------------------------------------------------


@pytest.mark.parametrize("phi", [0.5, 0.7, 0.9, 0.95])
@pytest.mark.parametrize("q", list(DEFAULT_HORIZONS))
def test_matches_the_ar1_closed_form(phi, q):
    """**The correctness guard.**

    For an AR(1) the true variance ratio is known exactly, so this checks the
    magnitude and not merely the sign. A bug in the ``m`` bias correction, or
    an off-by-one in the overlapping differences, changes the value by a few
    percent while leaving every sign-based test passing.

    Averaging over independent paths rather than testing one: a single
    realisation has sampling error of a few percent at these sample sizes, so
    a per-path tolerance loose enough to pass would be too loose to catch the
    bias it is meant to catch. The mean over 40 paths has a standard error
    small enough that a 3% tolerance is a real constraint.
    """
    expected = theoretical_ar1_variance_ratio(phi, q)
    estimates = [
        variance_ratio(ar1(4000, phi=phi, seed=1000 + s), q) for s in range(40)
    ]
    measured = float(np.mean(estimates))

    assert measured == pytest.approx(expected, rel=0.03), (
        f"phi={phi}, q={q}: measured {measured:.4f} vs closed form "
        f"{expected:.4f}; check the overlapping-window bias correction"
    )


def test_closed_form_tends_to_one_as_phi_approaches_a_unit_root():
    """A near-unit-root AR(1) is nearly a random walk, and the form must say so.

    This is the boundary where the half-life estimator misbehaves, so it is
    worth pinning that the theory used to validate this module behaves
    correctly there.
    """
    for q in DEFAULT_HORIZONS:
        assert theoretical_ar1_variance_ratio(0.9999, q) == pytest.approx(1.0, abs=1e-3)
        assert theoretical_ar1_variance_ratio(0.5, 1) == pytest.approx(1.0, abs=1e-12)


def test_closed_form_is_below_one_for_every_stationary_ar1():
    """Stationarity implies mean reversion implies VR < 1, at every horizon."""
    for phi in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        for q in DEFAULT_HORIZONS:
            assert theoretical_ar1_variance_ratio(phi, q) < 1.0


# --------------------------------------------------------------------------
# Direction: reverting, trending
# --------------------------------------------------------------------------


def test_a_strongly_mean_reverting_series_is_rejected_with_vr_below_one():
    """The positive control. A fast AR(1) must reject, and in the right direction.

    A test that never rejects anything would pass every calibration check
    above, so this is the complement that establishes the test has power.
    """
    series = ar1(2000, phi=0.7, seed=3)
    for result in variance_ratio_test(series):
        assert result.variance_ratio < 1.0
        assert result.p_robust < 0.01, (
            f"q={result.horizon}: failed to reject on a phi=0.7 AR(1), which "
            f"has a half-life under 2 bars (p={result.p_robust:.4f})"
        )
        assert result.rejects_random_walk()
        assert result.interpretation() == "mean-reverting"


def test_a_trending_series_gives_a_variance_ratio_above_one():
    """Positively autocorrelated increments compound, so VR must exceed 1.

    Built as a random walk plus a momentum term rather than as a deterministic
    trend: a straight line would give a large VR for the trivial reason that
    its long-horizon variance is quadratic, and would not test that the
    statistic responds to increment autocorrelation.
    """
    rng = np.random.default_rng(11)
    n = 2000
    increments = np.empty(n)
    increments[0] = rng.normal()
    for t in range(1, n):
        # AR(1) in the *increments* — positive persistence in returns.
        increments[t] = 0.4 * increments[t - 1] + rng.normal()
    series = np.cumsum(increments)

    for result in variance_ratio_test(series):
        assert result.variance_ratio > 1.0, (
            f"q={result.horizon}: VR={result.variance_ratio:.3f} on a trending "
            f"series"
        )
        assert result.p_robust < 0.01
        assert result.interpretation() == "trending"


def test_a_cointegrated_spread_is_rejected_but_its_legs_are_not():
    """The whole thesis of the repository, expressed as a variance ratio.

    ``generate_cointegrated`` builds two I(1) legs whose difference is I(0) by
    construction. The test must reject the random-walk null on the spread and
    fail to reject on either leg — that asymmetry is what cointegration *is*,
    and a statistic that rejected on the legs too would be detecting something
    other than mean reversion.
    """
    spec = CointegratedSpec(
        n_bars=2500, beta=1.0, half_life=15.0, sigma_spread=0.04, sigma_common=0.01
    )
    prices = generate_cointegrated(spec, seed=17)
    log_a = np.log(prices["price_a"].to_numpy())
    log_b = np.log(prices["price_b"].to_numpy())

    spread_results = variance_ratio_test(log_a - spec.beta * log_b)
    for result in spread_results:
        assert result.variance_ratio < 1.0

    # Rejection is required at the long horizons, not at q=2. With a 15-bar
    # half-life the true VR(2) is only about 0.977 — a 2% departure that the
    # test genuinely lacks the power to resolve, since q=2 sees just the
    # first-lag autocorrelation. Demanding a rejection there would be
    # demanding the test detect something it cannot, and this power gap is
    # precisely the reason DEFAULT_HORIZONS spans a range instead of using
    # one horizon.
    assert spread_results[-1].rejects_random_walk(), (
        f"q={spread_results[-1].horizon}: failed to reject on a spread that is "
        f"stationary by construction (p={spread_results[-1].p_robust:.4f})"
    )
    assert spread_results[-1].variance_ratio < 0.8, (
        "a 15-bar half-life should push the long-horizon ratio well below 1"
    )
    # And power must increase with the horizon, which is the mechanism.
    assert spread_results[-1].p_robust < spread_results[0].p_robust

    for name, leg in (("log_a", log_a), ("log_b", log_b)):
        for result in variance_ratio_test(leg):
            assert not result.rejects_random_walk(), (
                f"{name} q={result.horizon}: rejected on an I(1) leg "
                f"(p={result.p_robust:.4f})"
            )


def test_a_shorter_half_life_moves_the_ratio_further_below_one():
    """Monotonicity: faster reversion means a smaller variance ratio.

    The statistic should order series by how strongly they revert, not merely
    flag them. Without this, a test that saturated at some floor would pass
    every threshold check above.
    """
    ratios = [
        variance_ratio(ar1(4000, phi=phi, seed=21), q=8)
        for phi in (0.5, 0.8, 0.95, 0.99)
    ]
    assert all(a < b for a, b in zip(ratios, ratios[1:])), (
        f"VR must rise toward 1 as reversion slows, got {ratios}"
    )


# --------------------------------------------------------------------------
# The two nulls
# --------------------------------------------------------------------------


def test_heteroskedasticity_widens_the_robust_standard_error():
    """**Why the robust variant exists.**

    On a random walk with volatility clustering, the homoskedastic z-statistic
    is inflated because its variance formula uses only ``q`` and the sample
    size — it cannot see that the increments are heteroskedastic. The robust
    one estimates the variance from the data and must therefore be smaller in
    absolute value on exactly this data.

    Getting this backwards would understate uncertainty on the fat-tailed
    series this repository actually trades, in the direction that manufactures
    a finding.
    """
    rng = np.random.default_rng(5)
    n = 3000
    # A random walk whose innovation scale itself follows a persistent
    # process: uncorrelated increments, strongly time-varying variance. The
    # null is true here; only the homoskedasticity assumption is violated.
    log_vol = np.empty(n)
    log_vol[0] = 0.0
    for t in range(1, n):
        log_vol[t] = 0.98 * log_vol[t - 1] + rng.normal(0.0, 0.2)
    series = np.cumsum(rng.normal(0.0, 1.0, size=n) * np.exp(log_vol))

    for result in variance_ratio_test(series):
        assert abs(result.z_robust) < abs(result.z_homoskedastic), (
            f"q={result.horizon}: robust |z|={abs(result.z_robust):.3f} is not "
            f"below homoskedastic |z|={abs(result.z_homoskedastic):.3f}; the "
            f"robust correction is not accounting for the volatility clustering"
        )
        assert result.p_robust > result.p_homoskedastic


def test_the_two_nulls_agree_on_homoskedastic_data():
    """The complement: with the assumption satisfied, the two must be close.

    If the robust statistic were systematically far from the homoskedastic one
    even on Gaussian IID increments, it would be paying a large power cost for
    nothing rather than correcting a real problem.
    """
    series = random_walk(4000, seed=31)
    for result in variance_ratio_test(series):
        assert result.z_robust == pytest.approx(result.z_homoskedastic, rel=0.30)


# --------------------------------------------------------------------------
# Structural properties and input handling
# --------------------------------------------------------------------------


def test_scale_and_location_invariance():
    """VR is a ratio of variances of the same series, so units cannot matter.

    A spread measured in log points and the same spread multiplied by 100 must
    give an identical statistic. This catches a normalisation accidentally
    applied to one variance and not the other.
    """
    series = ar1(1500, phi=0.85, seed=13)
    for q in DEFAULT_HORIZONS:
        base = variance_ratio(series, q)
        assert variance_ratio(series * 100.0, q) == pytest.approx(base, rel=1e-10)
        assert variance_ratio(series + 500.0, q) == pytest.approx(base, rel=1e-10)


def test_drift_does_not_change_the_verdict():
    """A random walk with drift is still a random walk.

    The estimator removes the drift via the endpoint estimator, so adding a
    deterministic trend must leave the ratio near 1. Without that subtraction
    the trend would inflate the long-horizon variance and the test would call
    every drifting series "trending" — and real log-price series drift.

    Checked as "drift changes nothing" rather than "the drifting series is not
    rejected": the comparison is against the same path without the trend, so
    the assertion isolates the effect of the drift from the sampling noise of
    the particular path. A seed that happens to reject at some horizon rejects
    identically with and without the trend, which is the actual claim.
    """
    series = random_walk(2000, seed=23)
    drifting = series + 0.05 * np.arange(series.size)

    for plain, drifted in zip(
        variance_ratio_test(series), variance_ratio_test(drifting)
    ):
        assert drifted.variance_ratio == pytest.approx(
            plain.variance_ratio, rel=1e-9
        ), (
            f"q={plain.horizon}: adding a deterministic drift changed VR from "
            f"{plain.variance_ratio:.6f} to {drifted.variance_ratio:.6f}; the "
            f"drift is not being removed"
        )
        assert drifted.rejects_random_walk() == plain.rejects_random_walk()


def test_result_is_frozen_and_exports_a_flat_dict():
    """Matches how PairStats and SignificanceResult behave, for CSV export."""
    result = variance_ratio_test(random_walk(500, seed=1), horizons=(4,))[0]

    with pytest.raises(Exception):
        result.variance_ratio = 0.5  # type: ignore[misc]

    exported = result.to_dict()
    assert set(exported) == {
        "horizon",
        "variance_ratio",
        "z_homoskedastic",
        "p_homoskedastic",
        "z_robust",
        "p_robust",
        "n_obs",
        "rejects_random_walk",
        "interpretation",
    }
    assert all(not isinstance(v, (list, dict, tuple)) for v in exported.values())
    assert exported["horizon"] == 4
    assert exported["n_obs"] == 500


def test_interpretation_thresholds():
    """The sign reading, including the band around 1.0."""
    assert interpret(0.5) == "mean-reverting"
    assert interpret(0.94) == "mean-reverting"
    assert interpret(1.0) == "random walk"
    assert interpret(0.98) == "random walk"
    assert interpret(1.03) == "random walk"
    assert interpret(1.5) == "trending"
    assert interpret(float("nan")) == "undefined"


def test_horizons_are_returned_in_the_order_requested():
    """Callers index these positionally when building report rows."""
    horizons = (3, 2, 9)
    results = variance_ratio_test(random_walk(600, seed=2), horizons=horizons)
    assert tuple(r.horizon for r in results) == horizons


def test_a_constant_series_returns_nan_rather_than_dividing_by_zero():
    """Degenerate input must not raise mid-sweep or return a fabricated 1.0."""
    assert np.isnan(variance_ratio(np.zeros(100), q=4))
    result = variance_ratio_test(np.zeros(100), horizons=(4,))[0]
    assert np.isnan(result.p_robust)
    assert not result.rejects_random_walk()
    assert result.interpretation() == "undefined"


def test_bad_input_is_rejected_loudly():
    """A variance ratio is plausible at any value, so a wrong one looks fine.

    That is the argument for raising rather than returning a best effort: there
    is no output value a caller could inspect and recognise as invalid.
    """
    series = random_walk(200, seed=4)

    with pytest.raises(ValueError, match="at least 2"):
        variance_ratio(series, q=1)
    with pytest.raises(ValueError, match="at least 2"):
        variance_ratio(series, q=0)
    with pytest.raises(ValueError, match=f"at least {MIN_OBSERVATIONS}"):
        variance_ratio(random_walk(MIN_OBSERVATIONS - 1, seed=4), q=2)
    with pytest.raises(ValueError, match="too short for horizon"):
        variance_ratio(random_walk(40, seed=4), q=50)
    with pytest.raises(ValueError, match="finite"):
        variance_ratio(np.concatenate([series[:-1], [np.nan]]), q=2)
    with pytest.raises(ValueError, match="finite"):
        variance_ratio(np.concatenate([series[:-1], [np.inf]]), q=2)


def test_accepts_a_python_list():
    """Callers pass pandas columns and lists; asarray must handle both."""
    values = list(random_walk(300, seed=6))
    assert variance_ratio(values, q=4) == pytest.approx(
        variance_ratio(np.asarray(values), q=4)
    )
