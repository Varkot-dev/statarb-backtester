"""Tests for the structural-stability layer.

A break detector is only worth having if it does two things: fire on a break
that is really there, and stay silent on one that is not. The second is the
harder guarantee and the more important one here — a detector that flags
everything would "explain" MA/V's loss while explaining nothing, so the
no-break controls carry as much weight in this file as the positive cases.

Ground truth is available because the data is generated. A series built with
beta = 1.3 for the first half and 1.9 for the second has a break at a known
bar; one built with a single beta throughout has none.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from statarb.stability import (
    StabilityReport,
    analyse_stability,
    cusum_test,
    rolling_cointegration,
)
from statarb.synth import CointegratedSpec, business_dates, generate_cointegrated


# --------------------------------------------------------------------------
# Ground-truth generators
# --------------------------------------------------------------------------


def _spread_path(
    n_bars: int, half_life: float, sigma_spread: float, rng: np.random.Generator
) -> np.ndarray:
    """A stationary AR(1) spread, initialised at its stationary distribution."""
    phi = float(np.exp(-np.log(2.0) / half_life))
    innovation_sd = float(sigma_spread * np.sqrt(1.0 - phi**2))
    spread = np.empty(n_bars)
    spread[0] = rng.normal(0.0, sigma_spread)
    innovations = rng.normal(0.0, innovation_sd, size=n_bars)
    for t in range(1, n_bars):
        spread[t] = phi * spread[t - 1] + innovations[t]
    return spread


def _frame(log_a: np.ndarray, log_b: np.ndarray, n_bars: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"price_a": np.exp(log_a), "price_b": np.exp(log_b)},
        index=business_dates(n_bars),
    )


def _cointegrated_with_drift(
    n_bars: int = 2000,
    drift_per_bar: float = 2e-4,
    break_at: int | None = None,
    beta: float = 1.3,
    half_life: float = 20.0,
    sigma_spread: float = 0.03,
    sigma_common: float = 0.01,
    seed: int = 11,
) -> tuple[pd.DataFrame, int]:
    """A pair whose equilibrium level drifts away after a known bar.

    **This is the positive control, and the choice of a drift rather than a jump
    is deliberate.** CUSUM accumulates forecast errors, so the alternative it
    genuinely has power against is a *sustained* departure from the old
    relationship — the equilibrium walking away rather than stepping away. That
    is also the realistic failure mode for a pair: a business mix diverging over
    quarters, not overnight.

    Against an abrupt jump the test is much weaker and its power is
    non-monotone in the break size, for the variance-masking reason set out in
    the module docstring. Building the positive control as a jump would
    therefore either produce a flaky test or, if tuned until it passed, would
    hide the limitation instead of documenting it. The blind spot gets its own
    explicit test below.

    The drift is applied to the spread's level, leaving both legs I(1) and the
    price path continuous, so there is no jump for a detector to fire on
    spuriously. Returns the price frame and the bar at which the drift began.
    """
    rng = np.random.default_rng(seed)
    break_at = n_bars // 2 if break_at is None else break_at

    common_trend = np.cumsum(rng.normal(0.0, sigma_common, size=n_bars))
    spread = _spread_path(n_bars, half_life, sigma_spread, rng)
    ramp = np.clip(np.arange(n_bars) - break_at, 0, None) * drift_per_bar

    log_a = np.log(100.0) + beta * common_trend + spread + ramp
    log_b = np.log(50.0) + common_trend
    return _frame(log_a, log_b, n_bars), break_at


def _decointegrating_pair(
    n_bars: int = 2400,
    break_at: int = 1200,
    beta: float = 1.3,
    half_life: float = 6.0,
    sigma_spread: float = 0.03,
    sigma_common: float = 0.01,
    seed: int = 11,
) -> tuple[pd.DataFrame, int]:
    """A pair whose spread stops mean-reverting partway through the sample.

    The AR(1) coefficient on the spread switches from ``exp(-ln 2 / half_life)``
    to exactly 1 at ``break_at``: before, the spread is stationary and the pair
    is cointegrated; after, it is a random walk and the pair is not. Both legs
    stay I(1) and the price path is continuous, so there is no jump to fire on.

    **This is the break the rolling Engle-Granger test is actually built to
    find**, which is why it is the generator used for it. A change in the hedge
    ratio alone leaves the spread stationary — just at a different scale — and
    an ADF test on it still rejects, so the rolling p-values barely move. What
    ADF answers is whether the spread has a unit root, and a relationship
    ceasing to mean-revert is what makes a pairs trade stop working: the
    position no longer has an exit.

    The half-life is deliberately fast. The module docstring shows that the
    rolling test's power is governed by the window-to-half-life ratio, so a
    slow-reverting spread would produce weak pre-break rejection and the
    contrast would be measuring the test's power rather than the injected break.
    """
    rng = np.random.default_rng(seed)
    common_trend = np.cumsum(rng.normal(0.0, sigma_common, size=n_bars))

    phi = float(np.exp(-np.log(2.0) / half_life))
    innovation_sd = float(sigma_spread * np.sqrt(1.0 - phi**2))
    spread = np.empty(n_bars)
    spread[0] = rng.normal(0.0, sigma_spread)
    innovations = rng.normal(0.0, innovation_sd, size=n_bars)
    for t in range(1, n_bars):
        coefficient = phi if t < break_at else 1.0
        spread[t] = coefficient * spread[t - 1] + innovations[t]

    log_a = np.log(100.0) + beta * common_trend + spread
    log_b = np.log(50.0) + common_trend
    return _frame(log_a, log_b, n_bars), break_at


def _cointegrated_with_beta_break(
    n_bars: int = 2000,
    beta_before: float = 1.3,
    beta_after: float = 1.9,
    break_at: int | None = None,
    half_life: float = 20.0,
    sigma_spread: float = 0.03,
    sigma_common: float = 0.01,
    seed: int = 11,
) -> tuple[pd.DataFrame, int]:
    """A pair whose hedge ratio jumps at a known bar — the abrupt-break case.

    The switch is applied to the *increments* of the common trend rather than to
    its level, so leg A's log price stays continuous across the break. Switching
    the level would insert a price jump that a detector could fire on without
    ever noticing the parameter change, which would make a test pass for the
    wrong reason. What is injected is a change in the relationship only.

    Used to document the CUSUM blind spot rather than to demonstrate detection.
    """
    rng = np.random.default_rng(seed)
    break_at = n_bars // 2 if break_at is None else break_at

    common_trend = np.cumsum(rng.normal(0.0, sigma_common, size=n_bars))
    spread = _spread_path(n_bars, half_life, sigma_spread, rng)

    betas = np.where(np.arange(n_bars) < break_at, beta_before, beta_after)
    scaled = np.concatenate([[0.0], betas[1:] * np.diff(common_trend)])

    log_a = np.log(100.0) + np.cumsum(scaled) + spread
    log_b = np.log(50.0) + common_trend
    return _frame(log_a, log_b, n_bars), break_at


def _stable_pair(
    n_bars: int = 2000, seed: int = 3, half_life: float = 20.0
) -> pd.DataFrame:
    """A cointegrated pair with a single constant hedge ratio — the control.

    Generated through :func:`statarb.synth.generate_cointegrated` rather than a
    local copy, so the null case here is the same construction the rest of the
    repository validates against.
    """
    spec = CointegratedSpec(
        n_bars=n_bars,
        beta=1.3,
        half_life=half_life,
        sigma_spread=0.03,
        sigma_common=0.01,
    )
    frame = generate_cointegrated(spec, seed=seed)
    return pd.DataFrame(
        {"price_a": frame["price_a"].to_numpy(), "price_b": frame["price_b"].to_numpy()},
        index=business_dates(n_bars),
    )


# --------------------------------------------------------------------------
# CUSUM: the injected break
# --------------------------------------------------------------------------


def _crossing_rate(generator, n_seeds: int = 40, **kwargs) -> float:
    """Fraction of independently seeded samples in which CUSUM crosses.

    Size and power are properties of the *procedure*, not of any one sample, so
    they are measured across seeds. Asserting on a single sample would pin
    whichever way that seed happened to fall and would say nothing about the
    test's calibration.
    """
    crossings = 0
    for seed in range(n_seeds):
        prices = generator(seed=seed, **kwargs)
        if isinstance(prices, tuple):
            prices = prices[0]
        crossings += bool(
            cusum_test(prices["price_a"], prices["price_b"])["crossed"]
        )
    return crossings / n_seeds


def test_cusum_holds_its_nominal_size_on_stable_pairs():
    """**The negative control, and the one that matters most.**

    A detector that fires on a stable relationship cannot be used to explain
    anything, because it would have fired regardless — and the naive levels
    specification does exactly that, at a 95-100% rate (see the module
    docstring). This is the guarantee that makes a positive result on real data
    mean something, so it is measured across seeds rather than asserted once.

    The bound is generous relative to the nominal 5% because the
    error-correction form reduces the residual autocorrelation without
    eliminating it, and because 40 samples put a wide band around any rate. It
    is still far below what a mis-specified or mis-aligned test would produce.
    """
    rate = _crossing_rate(_stable_pair, n_seeds=40, n_bars=1500)
    assert rate <= 0.20, (
        f"false-positive rate {rate:.2f} on stable pairs is far above nominal; "
        f"the specification or the boundary alignment is wrong"
    )


def test_cusum_size_holds_across_reversion_speeds():
    """The size must not depend on the half-life.

    This is the specific failure of the levels specification: the faster the
    spread reverts the more autocorrelated its recursive residuals are, so a
    mis-specified test's false-positive rate varies with a parameter that has
    nothing to do with stability. Constant size across half-lives is evidence
    the autocorrelation has actually been absorbed.
    """
    rates = {
        half_life: _crossing_rate(
            _stable_pair, n_seeds=25, n_bars=1200, half_life=half_life
        )
        for half_life in (5.0, 20.0, 60.0)
    }
    assert all(rate <= 0.24 for rate in rates.values()), (
        f"size varies with reversion speed, so persistence is leaking into the "
        f"test: {rates}"
    )


def test_cusum_detects_sustained_equilibrium_drift():
    """**The positive control.** Power against the alternative CUSUM is built for.

    A relationship whose equilibrium walks away is what accumulating forecast
    errors detects. The rate here is far from 1.0 — this test is genuinely
    low-powered, as the module docstring quantifies — so the assertion is that
    drift is detected materially more often than the no-drift baseline, not
    that it is always detected.
    """
    baseline = _crossing_rate(_cointegrated_with_drift, n_seeds=40, drift_per_bar=0.0)
    drifting = _crossing_rate(_cointegrated_with_drift, n_seeds=40, drift_per_bar=2e-4)
    assert drifting > baseline + 0.2, (
        f"sustained drift must be detected more often than no drift: "
        f"{drifting:.2f} vs baseline {baseline:.2f}"
    )


def test_cusum_power_rises_with_the_size_of_the_drift():
    """Monotonicity against the alternative it has power against.

    Checked at the endpoints rather than pairwise across a fine grid, because
    Monte Carlo rates at 40 samples carry enough noise that adjacent points can
    invert without anything being wrong.
    """
    small = _crossing_rate(_cointegrated_with_drift, n_seeds=40, drift_per_bar=2e-5)
    large = _crossing_rate(_cointegrated_with_drift, n_seeds=40, drift_per_bar=2e-4)
    assert large > small, (
        f"a larger drift must be more detectable: {large:.2f} vs {small:.2f}"
    )


def test_cusum_is_weak_against_large_abrupt_breaks():
    """**The documented blind spot, pinned so it cannot be forgotten.**

    Against an abrupt jump the power is non-monotone: a large break inflates the
    full-sample variance that standardises the recursive residuals, so it widens
    its own denominator and masks itself. A very large hedge-ratio jump is
    therefore detected *less* often than a moderate one.

    This is asserted rather than merely written down because it bounds every
    conclusion drawn from a non-crossing, including the ones this repository
    draws about real pairs. If a future change made the test uniformly powerful,
    this assertion would fail and the module docstring would need rewriting —
    which is the correct outcome, not a nuisance.
    """
    moderate = _crossing_rate(
        _cointegrated_with_beta_break, n_seeds=40, beta_after=1.9
    )
    extreme = _crossing_rate(
        _cointegrated_with_beta_break, n_seeds=40, beta_after=3.5
    )
    assert extreme < moderate, (
        f"the variance-masking effect should make an extreme break less "
        f"detectable than a moderate one: {extreme:.2f} vs {moderate:.2f}"
    )
    assert extreme < 0.2, (
        f"an extreme abrupt break is expected to be largely invisible to "
        f"mean-CUSUM; got {extreme:.2f}"
    )


def test_cusum_detections_cluster_after_the_injected_break():
    """Detections must concentrate after the break, lagging it.

    CUSUM accumulates, so the boundary is breached some distance past the
    change. The assertion is distributional rather than absolute: a few
    detections land before the break because the test has a non-zero size and
    fires occasionally on nothing at all — the no-drift baseline measured here
    is the control that separates those from a genuine timing error. What must
    hold is that drift produces detections *and* that they sit mostly after the
    change, which is what would break if the two-step index offset in
    ``cusum_test`` were wrong.
    """
    break_at = 800
    before, after = 0, 0
    for seed in range(40):
        prices, _ = _cointegrated_with_drift(
            n_bars=2000, break_at=break_at, drift_per_bar=3e-4, seed=seed
        )
        result = cusum_test(prices["price_a"], prices["price_b"])
        if result["crossed"]:
            if result["break_index"] > break_at:
                after += 1
            else:
                before += 1

    assert after + before > 0, "no seed crossed; cannot check detection timing"
    assert after > before, (
        f"detections should follow the break: {after} after vs {before} before"
    )


def test_cusum_path_index_maps_back_to_the_input_bars():
    """The reported break index must address the *original* price series.

    The regression is run on differenced data and consumes further observations
    starting the recursion, so the path is shorter than the input and offset
    from it. If that offset were dropped or applied twice, every break this
    module reports would be dated to the wrong bar while still looking entirely
    plausible. This checks the mapping directly rather than inferring it.
    """
    prices = _stable_pair(n_bars=600)
    result = cusum_test(prices["price_a"], prices["price_b"])
    path = result["path"]

    labels = [d.strftime("%Y-%m-%d") for d in prices.index]
    for position in (0, len(path) // 2, len(path) - 1):
        bar = int(path["index"].iloc[position])
        assert path["date"].iloc[position] == labels[bar], (
            f"path row {position} claims bar {bar} but is dated "
            f"{path['date'].iloc[position]}, while bar {bar} is {labels[bar]}"
        )


def test_cusum_break_date_is_reported_from_the_index():
    """A break at bar 1,432 is much less useful than one at a date."""
    for seed in range(30):
        prices, _ = _cointegrated_with_drift(drift_per_bar=3e-4, seed=seed)
        result = cusum_test(prices["price_a"], prices["price_b"])
        if result["crossed"]:
            parsed = pd.Timestamp(result["break_date"])
            assert prices.index[0] <= parsed <= prices.index[-1]
            assert prices.index[result["break_index"]] == parsed
            return
    pytest.fail("no seed crossed; cannot check date reporting")


def test_cusum_path_is_aligned_and_complete():
    """The returned path must be plottable without further alignment work."""
    prices = _stable_pair(n_bars=800)
    path = cusum_test(prices["price_a"], prices["price_b"])["path"]

    assert list(path.columns) == [
        "index",
        "date",
        "cusum",
        "lower_bound",
        "upper_bound",
        "excursion",
    ]
    # The path must end on the last bar of the input and be dated from the
    # input's own index. Where it *starts* depends on how many observations the
    # error-correction regression consumes (one to differencing, three to the
    # initial parameter estimates), so it is checked as a small positive offset
    # rather than pinned to a literal that would break if the design changed.
    assert 0 < path["index"].iloc[0] <= 6
    assert path["index"].iloc[-1] == len(prices) - 1
    assert path["date"].iloc[-1] == prices.index[-1].strftime("%Y-%m-%d")
    assert len(path) == len(prices) - path["index"].iloc[0]
    assert np.all(path["lower_bound"] < path["upper_bound"])
    # The boundaries widen with the sample, which is what makes the test a
    # single statement about the whole path rather than a pointwise one.
    assert path["upper_bound"].iloc[-1] > path["upper_bound"].iloc[0]


def test_cusum_rejects_an_untabulated_confidence_level():
    """Brown-Durbin-Evans tabulated three levels; a fourth is not available.

    statsmodels falls through silently on an unsupported value rather than
    raising, so an unchecked 0.975 would return boundaries for some other level
    and the reported confidence would be a fiction.
    """
    prices = _stable_pair(n_bars=300)
    with pytest.raises(ValueError, match="0.9, 0.95, 0.99"):
        cusum_test(prices["price_a"], prices["price_b"], cusum_alpha=0.975)


# --------------------------------------------------------------------------
# Rolling cointegration
# --------------------------------------------------------------------------


def test_rolling_pvalues_degrade_around_an_injected_break():
    """Windows straddling the break must show weaker evidence than clean ones.

    This is the trajectory claim: a stable relationship has low p-values
    throughout, a broken one has a regime of low and a regime of high. Compared
    against the pre-break windows rather than against a fixed threshold, since
    the absolute level depends on the noise level chosen here.
    """
    prices, break_at = _decointegrating_pair()
    rolling = rolling_cointegration(prices["price_a"], prices["price_b"])
    assert len(rolling) > 0

    clean = rolling[rolling["end_index"] < break_at]
    post = rolling[rolling["start_index"] >= break_at]
    assert len(clean) > 0 and len(post) > 0, "need both regimes to compare"

    assert post["eg_pvalue"].mean() > clean["eg_pvalue"].mean() + 0.1, (
        f"windows after the break should show clearly weaker evidence: "
        f"{post['eg_pvalue'].mean():.4f} vs {clean['eg_pvalue'].mean():.4f}"
    )
    assert (clean["eg_pvalue"] < 0.05).mean() > (post["eg_pvalue"] < 0.05).mean(), (
        "the rejection rate must fall across the break"
    )


def test_rolling_hedge_ratio_recovers_the_true_beta_on_a_stable_pair():
    """Every window should recover the beta the data was built with.

    If the rolling estimator were biased, a stable pair would show drifting
    hedge ratios and the whole diagnostic would report instability that is
    really estimator error.
    """
    prices = _stable_pair()
    rolling = rolling_cointegration(prices["price_a"], prices["price_b"])
    assert rolling["hedge_ratio"].mean() == pytest.approx(1.3, abs=0.15)
    assert rolling["hedge_ratio"].std() < 0.2


def test_rolling_returns_an_empty_frame_when_the_sample_is_too_short():
    """A window computed on fewer bars than requested is not comparable.

    Returning an empty frame rather than silently shortening the window is the
    honest outcome — the alternative reports a statistic under a window the
    caller did not ask for.
    """
    prices = _stable_pair(n_bars=100)
    rolling = rolling_cointegration(prices["price_a"], prices["price_b"], window=252)
    assert len(rolling) == 0
    assert "eg_pvalue" in rolling.columns


def test_rolling_step_controls_the_window_count():
    """Overlapping windows carry little independent information per bar."""
    prices = _stable_pair(n_bars=1000)
    monthly = rolling_cointegration(prices["price_a"], prices["price_b"], step=21)
    weekly = rolling_cointegration(prices["price_a"], prices["price_b"], step=5)
    assert len(weekly) > len(monthly)


def test_rolling_rejects_degenerate_arguments():
    """A 5-bar window has no power; failing loudly beats reporting noise."""
    prices = _stable_pair(n_bars=400)
    with pytest.raises(ValueError, match="at least 30 bars"):
        rolling_cointegration(prices["price_a"], prices["price_b"], window=5)
    with pytest.raises(ValueError, match="at least 1 bar"):
        rolling_cointegration(prices["price_a"], prices["price_b"], step=0)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_non_positive_prices_are_rejected_not_dropped():
    """Dropping bad bars would misdate every break that follows them.

    Same reasoning as ``analyse_pair``: a non-positive price is a data problem,
    and silently shortening the sample hides it. Here it would also shift the
    CUSUM path relative to the date index.
    """
    prices = _stable_pair(n_bars=300).copy()
    prices.iloc[100, 0] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        cusum_test(prices["price_a"], prices["price_b"])


def test_misaligned_series_are_rejected():
    prices = _stable_pair(n_bars=300)
    with pytest.raises(ValueError, match="lengths differ"):
        cusum_test(prices["price_a"], prices["price_b"].iloc[:-5])


def test_a_sample_too_short_for_recursion_is_rejected():
    prices = _stable_pair(n_bars=300).iloc[:20]
    with pytest.raises(ValueError, match="at least 30 observations"):
        cusum_test(prices["price_a"], prices["price_b"])


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def test_report_flags_a_broken_pair_and_clears_a_stable_one():
    """End to end, on both ground truths. The headline behaviour.

    The broken case is judged on ``is_stable`` rather than on ``cusum_crossed``
    alone, which is the point of combining the two criteria: the rolling test
    catches a loss of mean-reversion that CUSUM is weak against, so the report
    as a whole reaches the right verdict even where one of its halves does not.

    Both directions are asserted on the same generator family, so a change that
    made the report fire on everything would fail the stable case and a change
    that made it fire on nothing would fail the broken one.
    """
    broken_prices, _ = _decointegrating_pair(seed=0)
    broken = analyse_stability(broken_prices["price_a"], broken_prices["price_b"])
    assert not broken.is_stable()
    assert broken.rolling_deterioration() > 0.4, (
        f"a spread that stops reverting should show clear deterioration, got "
        f"{broken.rolling_deterioration():.2f}"
    )

    stable_prices = _stable_pair(half_life=6.0)
    stable = analyse_stability(stable_prices["price_a"], stable_prices["price_b"])
    assert stable.is_stable()
    assert stable.hedge_ratio_drift() < 0.2


def test_report_is_frozen():
    """Frozen like ``PairStats`` and ``SignificanceResult``: a report is a record
    of what was measured, and mutating it after the fact would decouple the
    fields from the data they describe."""
    prices = _stable_pair(n_bars=400)
    report = analyse_stability(prices["price_a"], prices["price_b"], window=252)
    assert isinstance(report, StabilityReport)
    with pytest.raises(Exception):
        report.cusum_crossed = True  # type: ignore[misc]


def test_report_to_dict_is_json_serialisable():
    """The export must survive ``json.dumps`` without a custom encoder.

    numpy scalars and numpy bools are not serialisable by the standard encoder,
    and they are what the underlying computations naturally produce — so this
    pins the conversion rather than trusting it.
    """
    prices, _ = _cointegrated_with_beta_break(n_bars=1200)
    report = analyse_stability(prices["price_a"], prices["price_b"], "MA", "V")

    payload = report.to_dict()
    encoded = json.dumps(payload)
    restored = json.loads(encoded)

    assert restored["name_a"] == "MA"
    assert restored["name_b"] == "V"
    assert isinstance(restored["cusum_crossed"], bool)
    assert isinstance(restored["is_stable"], bool)
    assert isinstance(restored["verdict"], str)
    for key in (
        "fraction_of_sample_cointegrated",
        "cusum_max_excursion",
        "hedge_ratio_drift",
    ):
        assert isinstance(restored[key], float)


def test_report_handles_a_sample_shorter_than_one_window():
    """CUSUM still applies when the rolling test does not.

    The rolling fields go NaN and ``n_windows`` is zero rather than the whole
    report failing, because the two tests have different data requirements and
    losing the one that still works would be a needless loss of information.
    """
    prices = _stable_pair(n_bars=200)
    report = analyse_stability(prices["price_a"], prices["price_b"], window=252)
    assert report.n_windows == 0
    assert np.isnan(report.fraction_of_sample_cointegrated)
    assert report.first_failure_date is None
    # The CUSUM half still ran.
    assert np.isfinite(report.cusum_max_excursion)
    assert json.dumps(report.to_dict())


def test_verdict_names_the_break_when_there_is_one():
    """The one-line reading must be specific enough to act on.

    When CUSUM crosses, the verdict must name the date and the hedge-ratio
    movement — a bare "unstable" would send the reader back to the raw fields.
    Seeds are scanned because the test's low power means not every sample
    crosses.
    """
    for seed in range(30):
        prices, _ = _cointegrated_with_drift(drift_per_bar=3e-4, seed=seed)
        report = analyse_stability(prices["price_a"], prices["price_b"])
        if report.cusum_crossed:
            verdict = report.verdict()
            assert "structural break detected" in verdict
            assert report.cusum_break_date in verdict
            break
    else:
        pytest.fail("no seed crossed; cannot check the break verdict")

    stable_prices = _stable_pair()
    stable = analyse_stability(stable_prices["price_a"], stable_prices["price_b"])
    assert "no structural break detected" in stable.verdict()
