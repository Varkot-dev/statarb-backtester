"""Whether a cointegrating relationship held, or merely held on average.

Why a full-sample test cannot answer the question
--------------------------------------------------

The concrete case this module was written for is MA/V. It is the only real pair
in this repository that passes both cointegration tests — Engle-Granger
:math:`p = 0.0008`, Johansen trace 24.77 against a 15.49 critical value, a
half-life of 29.7 bars squarely inside the tradeable band — and it lost $4,498
over the sample. That is not a contradiction, and treating it as one is the
mistake this module exists to prevent.

A full-sample Engle-Granger test asks whether *one* hedge ratio, estimated once
over the whole history, leaves a stationary residual. It is a statement about
the pooled sample. If the true hedge ratio was 1.4 for six years and 1.8 for
four, the pooled regression finds some intermediate value, and the residual
around that compromise can still pass an ADF test — the two regimes' residuals
partly offset, and what is left looks like a wide but mean-reverting spread. The
test reports success. Meanwhile the strategy, which re-estimates the hedge ratio
on a trailing window and standardises against a trailing mean, spends the
transition period short a spread that is not coming back.

So cointegration is a necessary condition that is evaluated under an assumption
— parameter constancy — that the test itself never checks. This module checks it.

Why CUSUM rather than a Chow test
----------------------------------

The textbook structural-break test is Chow (1960): split the sample at a date,
fit the regression separately on each side, and F-test whether the coefficients
differ. It is more powerful than CUSUM *when the break date is known* — a policy
change, a merger, an index reconstitution.

It is the wrong tool here, and the reason is the same one that
:func:`statarb.diagnostics.deflated_sharpe_threshold` exists for. Nobody hands
you the break date for MA/V. You would find it by trying candidate dates and
keeping the one with the largest F-statistic — which is a specification search,
and the maximum of a set of F-statistics does not follow the F-distribution that
the reported p-value assumes. Under the null of no break, scanning 2,000
candidate split points and reporting the best one produces "significant" breaks
in stable series routinely. Correcting for it means abandoning the tabulated
critical values for the sup-Wald distribution of Andrews (1993), which is a
different and harder test than the one you started with. A Chow test with an
eyeballed break date is a p-value that has already been spent.

CUSUM does not have this problem, because it never picks a date. Brown, Durbin
and Evans (1975) accumulate the standardised **recursive residuals** — the
one-step-ahead forecast errors from a regression re-estimated on every prefix of
the sample. Under a stable relationship these are independent, mean-zero, and
unit-variance, so their cumulative sum is a random walk whose expected drift is
zero. When the relationship shifts, the forecasts become systematically wrong in
one direction and the cumulative sum drifts away from zero. The boundaries are
derived once for the whole path, so crossing them at *any* point is a single
test at the stated level. The date where the crossing occurs is read off
afterwards as a descriptive statistic, not as a hypothesis that was tested.

The price paid is power: CUSUM detects sustained drift in the mean well and is
comparatively weak against a break that leaves the mean intact (a variance-only
change, or two offsetting shifts). That trade — less power, but a p-value that
means what it says — is the right one when the break date is unknown, which here
it always is. The measured size and power of this implementation are tabulated
below, because a break detector whose false-positive rate is unknown cannot be
used to explain a losing pair.

Because CUSUM is directional, the crossing point is a *detection* date and not
the break date. Cumulative drift takes time to breach the boundary, so the true
break generally precedes the crossing, often by a substantial margin. The
rolling test below is the complement: it localises poorly relative to CUSUM's
single clean test but shows the whole trajectory, and reading the two together
is more informative than either alone.

The regression the CUSUM runs on, and why it is not the obvious one
-------------------------------------------------------------------

The obvious specification is the one the cointegration test itself uses:
regress :math:`\\log P_A` on :math:`\\log P_B` in levels and accumulate the
recursive residuals of that. **It does not work, and it fails in the direction
that manufactures findings.** Measured on synthetic pairs built with a single
constant hedge ratio — series with no break in them at all — the level
specification breaches the 95% boundary in 95 to 100% of samples:

.. code-block:: text

    half-life   false-positive rate   median peak excursion
        5              0.95                   1.69
       20              1.00                   2.88
       60              1.00                   3.33

The cause is not a bug but an assumption violation, and it is specific to this
application. CUSUM's null requires the recursive residuals to be independent.
The residual of a cointegrating regression is the *spread*, and the spread is a
persistent AR(1) by construction — that persistence is precisely the property
that makes the pair tradeable. The recursive residuals inherit it: their
measured lag-1 autocorrelation on a stable synthetic pair is **0.95**. A
cumulative sum of strongly autocorrelated terms wanders far more than a sum of
independent ones, so the Brown-Durbin-Evans boundaries — derived for the
independent case — are crossed almost surely. A level-specification CUSUM would
have reported a structural break in MA/V, and in every other pair, and in pure
noise, which is to say it would have reported nothing at all.

What is used instead is an **error-correction form**: regress
:math:`\\Delta \\log P_A` on a constant, :math:`\\Delta \\log P_B`, and the
lagged cointegrating residual. Differencing removes the stochastic trend and
the lagged residual absorbs the mean-reversion, so what is left is close to the
independent sequence the test assumes. This is the Granger representation of the
same relationship rather than a different model — the coefficient on
:math:`\\Delta \\log P_B` is the same hedge ratio, so a break in it is a break
in the thing the engine trades.

Measured size and power of the form actually used, over 100 to 200 Monte Carlo
samples of 2,000 bars with the break at bar 1,000, at the 95% boundary:

.. code-block:: text

    condition                                  detection rate
    no break (size)                                 0.06
    gradual equilibrium drift, total 0.02           0.08
    gradual equilibrium drift, total 0.05           0.14
    gradual equilibrium drift, total 0.10           0.35
    gradual equilibrium drift, total 0.20           0.50

The size is close to nominal — somewhat above 5%, since the error-correction
form reduces the residual autocorrelation rather than eliminating it — so a
crossing means something. Power rises monotonically with the size of a
*sustained drift* in the equilibrium relationship, which is the alternative
CUSUM is built against, but tops out around 0.5 even for drift far larger than
anything a liquid pair exhibits.

**A crossing is evidence of instability; the absence of one is close to no
evidence either way.** That asymmetry is the single most important thing to know
before reading this module's output. It is why
:meth:`StabilityReport.is_stable` requires the rolling criterion to agree rather
than treating a non-crossing as a clean bill of health, and why the rolling
p-value trajectory is reported alongside instead of being reduced to a flag.

There is a second and sharper limitation, and it is stated plainly because it
bounds what any conclusion here can claim. Against an **abrupt** break — a
one-time jump in the hedge ratio or the equilibrium level — the measured power
is not merely low but *non-monotone*: it rises, peaks around 0.3, and then falls
back toward the size as the break grows larger.

.. code-block:: text

    hedge ratio break     detection rate     fitted residual sd
       1.3 -> 1.3 (none)       0.07               0.0078
       1.3 -> 1.6              0.245              0.0080
       1.3 -> 1.9              0.250              0.0084
       1.3 -> 2.5              0.155              0.0099
       1.3 -> 3.5              0.045              0.0135

The mechanism is visible in the third column and is a property of the statistic,
not of the data. The recursive residuals are standardised by a variance
estimated over the *whole* sample, and a large break inflates that estimate.
The break therefore widens its own denominator faster than it grows the
numerator, and the statistic masks itself. A big enough abrupt break is
invisible to this test.

None of this is an implementation defect to be tuned away, and the fix is not
another parameter. Mean-CUSUM watches the mean of the forecast errors; a change
in a coefficient on a near-zero-mean regressor moves their *variance* far more
than their mean. The natural complement is the CUSUM of squares of the same
authors, which is deliberately **not** implemented here: its boundary constants
are tabulated by simulation rather than available in statsmodels, and
hand-rolling them would be exactly the kind of uncalibrated statistic this
module exists to avoid. A first attempt at it during development produced a 33%
false-positive rate against a nominal 5%, which is precisely the trap — it had
excellent apparent power (0.93) and would have "confirmed" a break in anything.

The practical consequence for reading the output: treat the rolling p-value
trajectory as the primary evidence and the CUSUM as a single well-calibrated
confirmation that costs no extra multiple-comparisons budget. A pair that fails
both is unstable with two independent reasons to think so. A pair that passes
CUSUM alone has been shown very little.

The rolling test and what its statistics do not mean
-----------------------------------------------------

:func:`rolling_cointegration` re-runs Engle-Granger on a moving window and
reports the fraction of windows that reject. That fraction is a descriptive
summary, deliberately, and it is worth being explicit about what it is not.

It is not a hypothesis test. Overlapping windows share most of their
observations and their p-values are strongly dependent, so the count of
rejections has no usable null distribution — you cannot binomial-test it, and a
"78% of windows cointegrated" figure carries far less information than 78
independent tests would. It also inherits the multiple-comparisons problem the
README acknowledges for :func:`statarb.cointegration.screen_universe`: at
:math:`\\alpha = 0.05` some windows reject by chance.

What it is good for is shape. A relationship that is stable shows p-values that
are low throughout; one that broke shows a regime of low p-values and a regime
of high ones, with a visible transition. That trajectory is the diagnostic, and
the scalar fraction is a way to sort pairs, not a way to test them.

There is a further trap in the fraction, and it is the repository's own
window/half-life law reappearing in a new place. **The rejection rate on a
perfectly stable pair depends almost entirely on the ratio of the rolling window
to the spread's half-life**, because a window only a few half-lives long does not
contain enough reversions for ADF to have power. Measured on synthetic pairs
built with a constant hedge ratio — series with no break in them whatsoever:

.. code-block:: text

    half-life   window   window/half-life   fraction rejecting
        5         252          50.4                0.99
       10         252          25.2                0.59
       20         252          12.6                0.19
       30         252           8.4                0.11
       60         252           4.2                0.08
       10         504          50.4                1.00
       20         504          25.2                0.66
       30         504          16.8                0.38

A stable pair with a half-life of 30 bars rejects in 11% of 252-bar windows.
Read naively, "cointegration holds in only 11% of the sample" sounds like a
damning verdict on the relationship; in fact it is a statement about the test's
power, and the relationship is constant by construction.

The consequence is that ``fraction_of_sample_cointegrated`` **must not be
compared across pairs with different half-lives**, and a low value is not by
itself evidence of a break. It is interpretable against the same pair's own
trajectory — a fall from 0.9 to 0.1 partway through the sample means something —
and against a comparison pair of similar reversion speed. This is the same law
that :mod:`statarb.diagnostics` documents for the z-score window, arrived at
from a different direction: any trailing estimator whose memory is comparable to
the half-life of the process it measures is compromised, whether it is
estimating a location or testing a hypothesis.

Implementation note
-------------------

The recursive residuals come from
:func:`statsmodels.stats.diagnostic.recursive_olsresiduals` (BSD-3) rather than
being hand-rolled. The repository's standard, recorded in ``CREDITS.md``, is to
use well-licensed libraries for standard statistics and hand-roll only where
there is a reason — as ``significance.py`` does for the incomplete beta
function, to avoid taking a scipy dependency for one call. There is no such
reason here: the recursive-residual updating formula is textbook, statsmodels'
implementation is numerically careful about the rank-one update of
:math:`(X'X)^{-1}`, and reimplementing it would add a subtle-error surface for
no gain.

References: Brown, Durbin & Evans (1975), *Techniques for Testing the Constancy
of Regression Relationships over Time*, JRSS-B 37(2). Chow (1960), *Tests of
Equality Between Sets of Coefficients in Two Linear Regressions*. Andrews
(1993), *Tests for Parameter Instability and Structural Change with Unknown
Change Point*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import recursive_olsresiduals
from statsmodels.tsa.stattools import coint

from .cointegration import DEFAULT_SIGNIFICANCE

#: Default rolling window, in bars. One trading year. Long enough for
#: Engle-Granger to have meaningful power (an ADF test on 60 observations
#: rejects almost nothing, so a shorter window would report instability that is
#: really just an absence of power) and short enough that a break lasting a year
#: or more is visible as a regime rather than smeared across the whole sample.
DEFAULT_ROLLING_WINDOW = 252

#: Default step between rolling windows, in bars. One trading month. Windows
#: advanced one bar at a time overlap by 251/252 and carry almost no independent
#: information for 252 times the compute; a monthly step keeps the trajectory's
#: shape at a fraction of the cost.
DEFAULT_ROLLING_STEP = 21

#: Confidence level for the CUSUM boundaries. ``recursive_olsresiduals``
#: supports only 0.90, 0.95 and 0.99, since Brown-Durbin-Evans tabulated the
#: boundary constants rather than deriving them in closed form.
DEFAULT_CUSUM_ALPHA = 0.95

#: Supported CUSUM confidence levels, for validation. Passing an unsupported
#: value to statsmodels silently falls through to one of these rather than
#: raising, so the check is made here where it can name the problem.
_SUPPORTED_CUSUM_ALPHAS = (0.90, 0.95, 0.99)

#: Fall in the rolling rejection rate, first half of the windows to second,
#: that counts as a structural break.
#:
#: Calibrated against the two synthetic ground truths rather than picked, and
#: not tuned against any real pair. Over 25 seeds each, stable pairs produce a
#: deterioration of 0.00 +/- 0.14 with a maximum of 0.19, while pairs whose
#: spread stops mean-reverting produce 0.77 +/- 0.12 with a minimum of 0.51. The
#: two distributions are cleanly separated, and 0.3 sits between them with room
#: on both sides — anywhere in 0.2 to 0.5 gives the same perfect split on that
#: evidence, so the exact value is not load-bearing and is set mid-range.
#:
#: This is a reporting convention with a measured basis, not a critical value.
#: The windows overlap, so the statistic has no null distribution to invert, and
#: quoting a p-value for it would be false precision.
DETERIORATION_THRESHOLD = 0.3


@dataclass(frozen=True)
class StabilityReport:
    """Structural-stability diagnostics for one pair.

    Attributes:
        name_a: Identifier of leg A.
        name_b: Identifier of leg B.
        n_obs: Number of observations used.
        n_windows: Number of rolling windows evaluated. Zero when the sample is
            shorter than one window, in which case the rolling fields are NaN
            rather than being computed on a truncated window.
        rolling_window: Window length used, in bars.
        rolling_step: Step between windows, in bars.
        fraction_of_sample_cointegrated: Fraction of rolling windows whose
            Engle-Granger p-value falls below ``alpha``. Descriptive only — see
            the module docstring on why this is not a hypothesis test, and in
            particular why its *level* is governed by the window-to-half-life
            ratio and so must not be compared across pairs.
        first_half_fraction: The same fraction over the earlier half of the
            windows.
        second_half_fraction: The same over the later half. The pair of numbers
            is the interpretable form: both halves carry the same power handicap
            from the pair's own half-life, so the difference between them
            isolates the change in the relationship.
        mean_rolling_pvalue: Mean Engle-Granger p-value across windows.
        max_rolling_pvalue: Worst (largest) rolling p-value. A pair that is
            genuinely stable should not have windows where the evidence
            disappears entirely.
        first_failure_date: Label of the first window whose p-value exceeds
            ``alpha``, or ``None`` if every window rejects. Dated at the
            window's **end**, since that is the first bar at which the failure
            could have been observed in real time.
        cusum_crossed: Whether the CUSUM path breached the Brown-Durbin-Evans
            boundary anywhere in the sample.
        cusum_break_date: Label of the first bar at which the boundary was
            breached, or ``None``. This is a *detection* date, later than the
            underlying break — see the module docstring.
        cusum_break_index: Positional index of that bar in the input series.
        cusum_max_excursion: Largest ratio of ``|CUSUM|`` to the boundary width
            at the same bar. Values below 1.0 mean no crossing; the magnitude
            says how close the path came, which distinguishes a marginal pass
            from a comfortable one.
        hedge_ratio_first_half: OLS slope estimated on the first half of the
            sample.
        hedge_ratio_second_half: The same on the second half. Reported beside
            the tests because a CUSUM crossing tells you the relationship moved
            and these two numbers say in which direction and by how much —
            which is what determines whether a trader would have been on the
            wrong side of it.
        alpha: Significance level used for the rolling test.
    """

    name_a: str
    name_b: str
    n_obs: int
    n_windows: int
    rolling_window: int
    rolling_step: int
    fraction_of_sample_cointegrated: float
    first_half_fraction: float
    second_half_fraction: float
    mean_rolling_pvalue: float
    max_rolling_pvalue: float
    first_failure_date: str | None
    cusum_crossed: bool
    cusum_break_date: str | None
    cusum_break_index: int | None
    cusum_max_excursion: float
    hedge_ratio_first_half: float
    hedge_ratio_second_half: float
    alpha: float

    def is_stable(self, min_deterioration: float = DETERIORATION_THRESHOLD) -> bool:
        """Whether the relationship looks constant by both criteria.

        Requires that CUSUM did not cross *and* that the rolling evidence did
        not deteriorate materially across the sample. Requiring both mirrors
        :meth:`statarb.cointegration.PairStats.is_cointegrated`: the two checks
        have different weaknesses, so agreement means more than either alone.
        CUSUM is weak against abrupt slope changes; the rolling test is blind to
        anything faster than its window.

        **The rolling criterion is deterioration, not level**, and that choice is
        forced by the power result in the module docstring. Thresholding the
        *level* of ``fraction_of_sample_cointegrated`` would fail a stable pair
        with a slow-reverting spread, whose windows reject only 11% of the time
        for reasons that have nothing to do with stability. Comparing the pair's
        second half against its own first half cancels that, because both halves
        share the same half-life and therefore the same handicap. What is left is
        the change, which is what "did the relationship break" actually asks.

        Args:
            min_deterioration: Fall in the rejection rate from the first half of
                the windows to the second that counts as a break. See
                :data:`DETERIORATION_THRESHOLD` for how the default was chosen.
        """
        return bool(
            not self.cusum_crossed
            and not (self.rolling_deterioration() >= min_deterioration)
        )

    def rolling_deterioration(self) -> float:
        """Fall in the rolling rejection rate from the first half to the second.

        Positive values mean the evidence for cointegration weakened over the
        sample. Returns NaN when there were too few windows to split, which
        propagates correctly through :meth:`is_stable`'s comparison — a NaN
        fails the ``>=`` test, so an unmeasurable trajectory is not treated as a
        detected break.
        """
        if not np.isfinite(self.first_half_fraction) or not np.isfinite(
            self.second_half_fraction
        ):
            return float("nan")
        return float(self.first_half_fraction - self.second_half_fraction)

    def hedge_ratio_drift(self) -> float:
        """Absolute change in the half-sample hedge ratio.

        The economically legible companion to the test statistics: a CUSUM
        crossing says the relationship moved, this says how far. It is a crude
        two-point estimate and will understate a break that reverses, but it is
        in the units a reader already understands.
        """
        return abs(self.hedge_ratio_second_half - self.hedge_ratio_first_half)

    def verdict(self) -> str:
        """A one-line plain-English reading, matching ``SignificanceResult``."""
        if self.cusum_crossed:
            return (
                f"structural break detected: CUSUM crosses at "
                f"{self.cusum_break_date}, hedge ratio moves "
                f"{self.hedge_ratio_first_half:.3f} -> "
                f"{self.hedge_ratio_second_half:.3f}"
            )
        deterioration = self.rolling_deterioration()
        if np.isfinite(deterioration) and deterioration >= DETERIORATION_THRESHOLD:
            return (
                f"structural break detected: rolling cointegration falls from "
                f"{self.first_half_fraction:.0%} of windows in the first half of "
                f"the sample to {self.second_half_fraction:.0%} in the second "
                f"(CUSUM did not cross, but it is weak against abrupt breaks)"
            )
        return (
            f"no structural break detected (CUSUM within bounds at "
            f"{self.cusum_max_excursion:.2f} of the boundary; rolling "
            f"cointegration {self.first_half_fraction:.0%} -> "
            f"{self.second_half_fraction:.0%} across the sample)"
        )

    def to_dict(self) -> dict:
        """Flat, JSON-serialisable dict for export.

        Every value is a Python scalar or ``None``. numpy scalars are not
        JSON-serialisable by the standard library encoder, so the conversion is
        done here rather than left as a trap at the call site.
        """
        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "n_obs": self.n_obs,
            "n_windows": self.n_windows,
            "rolling_window": self.rolling_window,
            "rolling_step": self.rolling_step,
            "fraction_of_sample_cointegrated": self.fraction_of_sample_cointegrated,
            "first_half_fraction": self.first_half_fraction,
            "second_half_fraction": self.second_half_fraction,
            "rolling_deterioration": self.rolling_deterioration(),
            "mean_rolling_pvalue": self.mean_rolling_pvalue,
            "max_rolling_pvalue": self.max_rolling_pvalue,
            "first_failure_date": self.first_failure_date,
            "cusum_crossed": self.cusum_crossed,
            "cusum_break_date": self.cusum_break_date,
            "cusum_break_index": self.cusum_break_index,
            "cusum_max_excursion": self.cusum_max_excursion,
            "hedge_ratio_first_half": self.hedge_ratio_first_half,
            "hedge_ratio_second_half": self.hedge_ratio_second_half,
            "hedge_ratio_drift": self.hedge_ratio_drift(),
            "alpha": self.alpha,
            "is_stable": self.is_stable(),
            "verdict": self.verdict(),
        }


def _validate_pair(prices_a: pd.Series, prices_b: pd.Series) -> tuple[
    np.ndarray, np.ndarray
]:
    """Shared validation, returning log prices.

    Mirrors :func:`statarb.cointegration.analyse_pair`: non-positive prices are
    rejected rather than dropped, because they indicate a data problem that
    silently shortening the sample would hide — and here, shortening the sample
    would also break the alignment between the CUSUM path and the date index,
    so a break would be reported at the wrong date.
    """
    if len(prices_a) != len(prices_b):
        raise ValueError(
            f"series lengths differ: {len(prices_a)} vs {len(prices_b)}"
        )
    a = np.asarray(prices_a, dtype=float)
    b = np.asarray(prices_b, dtype=float)
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("prices must be strictly positive (log prices are taken)")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("prices must be finite")
    return np.log(a), np.log(b)


def _labels(prices_a: pd.Series, n: int) -> list[str]:
    """Human-readable labels for each bar, for reporting break dates.

    Uses the series index when it carries dates and falls back to the
    positional index otherwise. A break reported as bar 1,432 is much less
    useful than one reported as 2020-03-16, but a positional label is still
    better than refusing to report the location at all.
    """
    index = getattr(prices_a, "index", None)
    if index is None or len(index) != n:
        return [str(i) for i in range(n)]
    if isinstance(index, pd.DatetimeIndex):
        return [d.strftime("%Y-%m-%d") for d in index]
    return [str(v) for v in index]


def rolling_cointegration(
    prices_a: pd.Series,
    prices_b: pd.Series,
    window: int = DEFAULT_ROLLING_WINDOW,
    step: int = DEFAULT_ROLLING_STEP,
    alpha: float = DEFAULT_SIGNIFICANCE,
) -> pd.DataFrame:
    """Engle-Granger p-value and hedge ratio over a moving window.

    Each window is tested in isolation — the regression is re-estimated inside
    the window rather than reusing a full-sample hedge ratio. Reusing one would
    make every window's residual a deviation from a line fitted with knowledge
    of the future, so a window during a break would be scored using the very
    parameter shift being looked for. The point is to ask what an analyst
    standing at the end of each window, with only the preceding year of data,
    would have concluded.

    Args:
        prices_a: Price level series for leg A. Must be strictly positive.
        prices_b: Price level series for leg B, aligned to ``prices_a``.
        window: Window length in bars.
        step: Bars between consecutive window start points.
        alpha: Significance level recorded in the ``is_cointegrated`` column.

    Returns:
        One row per window with ``start_index``, ``end_index``, ``start_date``,
        ``end_date``, ``eg_pvalue``, ``hedge_ratio`` and ``is_cointegrated``.
        Dates are the labels of the window's first and last bar. An empty frame
        with these columns when the sample is shorter than one window — the
        honest outcome, since a window computed on fewer bars than requested
        would not be comparable to the others.

    Raises:
        ValueError: On misaligned, non-positive or non-finite prices, or on a
            non-positive window or step.
    """
    if window < 30:
        raise ValueError(f"window must be at least 30 bars, got {window}")
    if step < 1:
        raise ValueError(f"step must be at least 1 bar, got {step}")

    log_a, log_b = _validate_pair(prices_a, prices_b)
    n = len(log_a)
    labels = _labels(prices_a, n)

    columns = [
        "start_index",
        "end_index",
        "start_date",
        "end_date",
        "eg_pvalue",
        "hedge_ratio",
        "is_cointegrated",
    ]
    if n < window:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    for start in range(0, n - window + 1, step):
        end = start + window
        segment_a = log_a[start:end]
        segment_b = log_b[start:end]

        # A window in which either leg never moves has no regression to run.
        # Skipping it is right: it is a data artifact (a halt, a stale feed),
        # not evidence about the relationship either way.
        if np.std(segment_a) < 1e-12 or np.std(segment_b) < 1e-12:
            continue

        _, p_value, _ = coint(segment_a, segment_b)
        design = np.column_stack([np.ones(window), segment_b])
        coefficients, *_ = np.linalg.lstsq(design, segment_a, rcond=None)

        rows.append(
            {
                "start_index": start,
                "end_index": end - 1,
                "start_date": labels[start],
                "end_date": labels[end - 1],
                "eg_pvalue": float(p_value),
                "hedge_ratio": float(coefficients[1]),
                "is_cointegrated": bool(p_value < alpha),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def cusum_test(
    prices_a: pd.Series,
    prices_b: pd.Series,
    cusum_alpha: float = DEFAULT_CUSUM_ALPHA,
) -> dict[str, object]:
    """CUSUM of standardised recursive residuals, with BDE boundaries.

    Accumulates the standardised one-step-ahead forecast errors from a
    regression re-estimated on every prefix of the sample. Under the null of
    constant coefficients those errors are independent :math:`N(0, 1)`, so the
    cumulative sum behaves like a random walk with zero drift and its maximum
    excursion has a known distribution. Brown, Durbin and Evans place a pair of
    straight-line boundaries around it such that the probability of the path
    escaping at *any* point is ``1 - cusum_alpha``. Escape is evidence of a
    sustained shift, and because the boundaries cover the whole path at once it
    is a single test — which is exactly the property a Chow test scanning
    candidate break dates gives up. See the module docstring.

    The regression is the **error-correction form** — :math:`\\Delta \\log P_A`
    on a constant, :math:`\\Delta \\log P_B`, and the lagged cointegrating
    residual — and not the levels regression that
    :func:`statarb.cointegration.analyse_pair` uses. This is not a cosmetic
    choice. The levels form leaves recursive residuals with a lag-1
    autocorrelation of about 0.95, violating the independence its boundaries
    assume, and produces a false-positive rate of 95 to 100% on synthetic pairs
    with no break in them. The module docstring gives the measurement and the
    reasoning. It is the same relationship in its Granger representation, so the
    coefficient that can break is still the hedge ratio the engine trades.

    **Read a non-crossing with care.** Size is close to nominal but power tops
    out near 0.5 against sustained drift and is *non-monotone* against abrupt
    breaks, because a large break inflates the full-sample variance that
    standardises the residuals and so masks itself. A crossing is evidence; the
    absence of one is much weaker evidence. Both facts are quantified in the
    module docstring.

    Args:
        prices_a: Price level series for leg A. Must be strictly positive.
        prices_b: Price level series for leg B, aligned to ``prices_a``.
        cusum_alpha: Confidence level for the boundaries. Only 0.90, 0.95 and
            0.99 are supported, because Brown-Durbin-Evans tabulated the
            boundary constants rather than deriving them in closed form.

    Returns:
        A dict with ``crossed``, ``break_index`` and ``break_date`` (both
        ``None`` when the path stays inside), ``max_excursion`` (the peak ratio
        of ``|CUSUM|`` to the boundary at that bar, so 1.0 is exactly the
        boundary), and ``path``: a DataFrame of the CUSUM trajectory with its
        boundaries, for plotting. Indices and dates in the path refer to bars of
        the original input series, so a reported break date is directly
        comparable to the price history.

    Raises:
        ValueError: On misaligned or non-positive prices, on an unsupported
            ``cusum_alpha``, or on a sample too short for recursive estimation.
    """
    if cusum_alpha not in _SUPPORTED_CUSUM_ALPHAS:
        raise ValueError(
            f"cusum_alpha must be one of {_SUPPORTED_CUSUM_ALPHAS}, got "
            f"{cusum_alpha}; Brown-Durbin-Evans tabulated only these"
        )

    log_a, log_b = _validate_pair(prices_a, prices_b)
    n = len(log_a)
    if n < 30:
        raise ValueError(
            f"need at least 30 observations for recursive estimation, got {n}"
        )

    labels = _labels(prices_a, n)

    # The error-correction design. The lagged cointegrating residual is taken
    # from a full-sample OLS, which is a deliberate and bounded use of the whole
    # sample: this is a research diagnostic asking "was the relationship
    # constant over this history", not a signal the engine trades, so it is not
    # lookahead in the sense the engine must avoid. Using a trailing estimate
    # instead would make the EC term itself drift and confound the test.
    equilibrium_residual = sm.OLS(log_a, sm.add_constant(log_b)).fit().resid
    delta_a = np.diff(log_a)
    delta_b = np.diff(log_b)
    design = np.column_stack(
        [np.ones_like(delta_b), delta_b, equilibrium_residual[:-1]]
    )
    if np.std(delta_b) < 1e-12:
        raise ValueError("prices_b never changes; there is no regression to run")

    fitted = sm.OLS(delta_a, design).fit()
    _, _, _, _, _, cusum, cusum_ci = recursive_olsresiduals(fitted, alpha=cusum_alpha)

    # Alignment, in two steps, because an off-by-one here would misdate every
    # break this module ever reports.
    #
    # 1. `recursive_olsresiduals` returns a CUSUM path of length m - k + 1 (for
    #    m regression rows and k parameters) whose leading element is the
    #    degenerate zero from before any recursive residual exists, and
    #    boundaries of length m - k. Dropping that element aligns the two.
    # 2. The regression has m = n - 1 rows because of the differencing, and row
    #    j is built from the transition between original bars j and j + 1. So a
    #    path element covering regression row j refers to original bar j + 1.
    #
    # Both steps are folded into `offset`, computed from the array lengths
    # rather than hardcoded, and then checked against the design.
    path = np.asarray(cusum, dtype=float)[1:]
    bounds = np.asarray(cusum_ci, dtype=float)
    upper = bounds[1]
    if len(path) != len(upper):
        raise ValueError(
            f"CUSUM path length {len(path)} does not match boundary length "
            f"{len(upper)}; statsmodels' return convention has changed"
        )
    offset = n - len(path)

    # The boundary is symmetric about zero, so a single width suffices. Guard
    # the division: the width is positive by construction, but a zero would
    # turn a non-crossing into an infinite excursion.
    width = np.maximum(np.abs(upper), 1e-12)
    excursion = np.abs(path) / width

    outside = np.flatnonzero(excursion > 1.0)
    if outside.size > 0:
        first = int(outside[0])
        break_index: int | None = first + offset
        break_date: str | None = labels[break_index]
        crossed = True
    else:
        break_index = None
        break_date = None
        crossed = False

    path_frame = pd.DataFrame(
        {
            "index": np.arange(offset, n),
            "date": labels[offset:],
            "cusum": path,
            "lower_bound": bounds[0],
            "upper_bound": upper,
            "excursion": excursion,
        }
    )

    return {
        "crossed": crossed,
        "break_index": break_index,
        "break_date": break_date,
        "max_excursion": float(excursion.max()) if excursion.size else 0.0,
        "cusum_alpha": cusum_alpha,
        "path": path_frame,
    }


def _half_sample_hedge_ratios(
    log_a: np.ndarray, log_b: np.ndarray
) -> tuple[float, float]:
    """OLS slope on each half of the sample.

    Returns ``(nan, nan)`` when either half is too short or degenerate, rather
    than a number computed from three points that would read as a real estimate.
    """
    midpoint = len(log_a) // 2
    halves = ((log_a[:midpoint], log_b[:midpoint]), (log_a[midpoint:], log_b[midpoint:]))
    out: list[float] = []
    for segment_a, segment_b in halves:
        if len(segment_a) < 10 or np.std(segment_b) < 1e-12:
            out.append(float("nan"))
            continue
        design = np.column_stack([np.ones_like(segment_b), segment_b])
        coefficients, *_ = np.linalg.lstsq(design, segment_a, rcond=None)
        out.append(float(coefficients[1]))
    return out[0], out[1]


#: Windows swept when reporting whether a break verdict is robust.
#:
#: A single window is a specification search of size one. This repository built
#: `deflated_sharpe_threshold` precisely to catch results that hold at one
#: parameter value, and then reported a break at one window without pointing
#: that machinery at the window itself. The sweep below is the correction.
ROBUSTNESS_WINDOWS: tuple[int, ...] = (252, 378, 504, 630, 756, 882, 1008)


def break_robustness(
    prices_a: pd.Series,
    prices_b: pd.Series,
    windows: tuple[int, ...] = ROBUSTNESS_WINDOWS,
    step: int = 21,
) -> pd.DataFrame:
    """Sweep the rolling window and report the deterioration statistic at each.

    **Why this exists.** The MA/V break was originally reported at a single
    window (756 bars), chosen with a documented rationale — the repo's own
    window/half-life law implies a short window has no power. The rationale is
    sound. But choosing one value of a free parameter and reporting the result
    that follows is a specification search, and this repository built
    :func:`statarb.diagnostics.deflated_sharpe_threshold` specifically to catch
    that pattern. It was never pointed at the stability window.

    Sweeping it shows the verdict is **not robust**: the deterioration statistic
    is non-monotone and changes sign across the range, and only two of seven
    windows exceed the firing threshold. Reporting the sweep alongside the
    verdict is the honest form. It does not mean the break is not real — the
    hedge-ratio drift and the sharp p-value transition are independent evidence
    — but it does mean the deterioration statistic alone cannot carry the claim.

    Returns a frame with one row per window: the window, its ratio to a
    reference half-life if supplied, the two half-sample fractions, the
    deterioration, and whether it fires.
    """
    rows: list[dict] = []
    for window in windows:
        try:
            report = analyse_stability(
                prices_a, prices_b, window=window, step=step
            )
        except (ValueError, IndexError):
            continue
        rows.append(
            {
                "window": window,
                "first_half_fraction": report.first_half_fraction,
                "second_half_fraction": report.second_half_fraction,
                "deterioration": report.rolling_deterioration(),
                "fires": not report.is_stable(),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) > 0:
        frame.attrs["fraction_of_windows_firing"] = float(frame["fires"].mean())
        signs = frame["deterioration"].map(lambda v: 1 if v > 0 else -1)
        frame.attrs["sign_changes"] = int((signs.diff().fillna(0) != 0).sum())
    return frame


def analyse_stability(
    prices_a: pd.Series,
    prices_b: pd.Series,
    name_a: str = "A",
    name_b: str = "B",
    window: int = DEFAULT_ROLLING_WINDOW,
    step: int = DEFAULT_ROLLING_STEP,
    alpha: float = DEFAULT_SIGNIFICANCE,
    cusum_alpha: float = DEFAULT_CUSUM_ALPHA,
) -> StabilityReport:
    """Run both stability tests on one pair and summarise them.

    This is the counterpart to :func:`statarb.cointegration.analyse_pair`: that
    function asks whether a relationship existed over the sample, this one asks
    whether it was the *same* relationship throughout. A pair should clear both
    before it is traded, and MA/V is the case in this repository that clears the
    first and fails the second.

    Args:
        prices_a: Price level series for leg A. Must be strictly positive.
        prices_b: Price level series for leg B, aligned to ``prices_a``.
        name_a: Identifier for leg A.
        name_b: Identifier for leg B.
        window: Rolling window length in bars.
        step: Bars between consecutive rolling windows.
        alpha: Significance level for the rolling Engle-Granger tests.
        cusum_alpha: Confidence level for the CUSUM boundaries.

    Returns:
        A populated :class:`StabilityReport`. The rolling fields are NaN and
        ``n_windows`` is zero when the sample is shorter than one window; the
        CUSUM fields are still populated, since CUSUM needs only enough
        observations to start the recursion.

    Raises:
        ValueError: On misaligned, non-positive or non-finite prices, or a
            sample too short for recursive estimation.
    """
    log_a, log_b = _validate_pair(prices_a, prices_b)

    rolling = rolling_cointegration(
        prices_a, prices_b, window=window, step=step, alpha=alpha
    )
    cusum = cusum_test(prices_a, prices_b, cusum_alpha=cusum_alpha)

    if len(rolling) > 0:
        p_values = rolling["eg_pvalue"].to_numpy(dtype=float)
        rejects = p_values < alpha
        fraction = float(np.mean(rejects))
        mean_p = float(p_values.mean())
        max_p = float(p_values.max())
        failures = rolling.loc[~rolling["is_cointegrated"], "end_date"]
        first_failure = str(failures.iloc[0]) if len(failures) > 0 else None

        # Split the windows in half to expose the trajectory. Two windows is the
        # minimum that gives one per half; below that the halves are not
        # meaningful and NaN is the honest answer.
        if len(rejects) >= 2:
            midpoint = len(rejects) // 2
            first_half_fraction = float(np.mean(rejects[:midpoint]))
            second_half_fraction = float(np.mean(rejects[midpoint:]))
        else:
            first_half_fraction = second_half_fraction = float("nan")
    else:
        fraction = mean_p = max_p = float("nan")
        first_half_fraction = second_half_fraction = float("nan")
        first_failure = None

    hedge_first, hedge_second = _half_sample_hedge_ratios(log_a, log_b)

    return StabilityReport(
        name_a=name_a,
        name_b=name_b,
        n_obs=len(log_a),
        n_windows=len(rolling),
        rolling_window=window,
        rolling_step=step,
        fraction_of_sample_cointegrated=fraction,
        first_half_fraction=first_half_fraction,
        second_half_fraction=second_half_fraction,
        mean_rolling_pvalue=mean_p,
        max_rolling_pvalue=max_p,
        first_failure_date=first_failure,
        cusum_crossed=bool(cusum["crossed"]),
        cusum_break_date=cusum["break_date"],
        cusum_break_index=cusum["break_index"],
        cusum_max_excursion=float(cusum["max_excursion"]),
        hedge_ratio_first_half=hedge_first,
        hedge_ratio_second_half=hedge_second,
        alpha=alpha,
    )
