"""The Lo-MacKinlay variance ratio test: a p-value on "this is a random walk".

Why this module exists
----------------------

:func:`statarb.cointegration.estimate_half_life` answers "on what timescale
does this spread revert" and, by its own documented warning, cannot answer "is
it mean-reverting at all". The estimator is biased downward on near-unit-root
series, so a genuine random walk comes back with a large but *finite*
half-life — around 350 bars on daily data in this repository's own negative
control — rather than the infinity the truth would imply. There is no
threshold on that number that separates reversion from noise, because the
statistic has no known distribution under the null of a random walk. It is a
point estimate with no error bar.

The variance ratio test supplies exactly the missing half. It is a *hypothesis
test*: under the null that the series is a random walk, the statistic has a
known asymptotic distribution, so the output is a p-value rather than a
magnitude. That is the complementarity, and it is the reason to add this
alongside the half-life rather than in place of it:

- **Half-life**: how fast, no significance. Biased toward finding reversion.
- **Variance ratio**: whether at all, with a p-value. No timescale.

Used together, a spread is worth trading only if the variance ratio test
rejects the random-walk null *and* the half-life falls in a horizon the engine
can hold. Either one alone admits a failure mode the other catches.

How the statistic works
-----------------------

Under a random walk, variance is linear in the sampling interval: the variance
of a :math:`q`-bar change is exactly :math:`q` times the variance of a one-bar
change, because the increments are uncorrelated and variances of independent
things add. The variance ratio is the sample version of that identity,

.. math:: VR(q) = \\frac{\\mathrm{Var}(X_t - X_{t-q})}{q \\cdot
          \\mathrm{Var}(X_t - X_{t-1})}

so the null predicts :math:`VR(q) = 1` at every horizon. Departures are
directly interpretable, which is the second reason to prefer this over a bare
unit-root test:

- :math:`VR < 1` — **mean-reverting.** Negatively autocorrelated increments
  partially cancel over long horizons, so the long-horizon variance grows more
  slowly than linearly. This is what a tradeable spread looks like.
- :math:`VR > 1` — **trending.** Positively autocorrelated increments compound.
- :math:`VR \\approx 1` — **random walk.** No exploitable structure.

Equivalently, :math:`VR(q) = 1 + 2\\sum_{k=1}^{q-1}(1 - k/q)\\rho_k`, a
triangularly-weighted sum of the increment autocorrelations. Reading it that
way explains why the test is run at several horizons: each :math:`q` weights a
different stretch of the autocorrelation function, and a spread that reverts
over twenty bars can look like a random walk at :math:`q = 2` while rejecting
clearly at :math:`q = 16`.

Why the heteroskedasticity-robust variant is the one to read
------------------------------------------------------------

Lo and MacKinlay give two null hypotheses. The first (their M1 statistic)
assumes homoskedastic IID increments. The second (M2) assumes only that
increments are uncorrelated, allowing arbitrary conditional heteroskedasticity
— which still implies :math:`VR = 1`, because uncorrelatedness is all the
random-walk variance identity needs.

**M2 is the statistic to trust here, and the difference is not academic.**
The homoskedastic standard error is derived under a normality-flavoured
assumption that this repository's own data violates badly: the NAV returns
have kurtosis around 14 against a normal value of 3 (documented in
``CREDITS.md``), and financial spreads exhibit volatility clustering as a
matter of course. Under those conditions the homoskedastic standard error is
too small, the z-statistic is inflated, and the test rejects the random-walk
null more often than its nominal 5% — finding mean reversion in series that
have none. That is the same direction of error the rest of this repository
exists to catch, so both are reported and the robust one is the one that
should drive a decision.

Where they disagree, the disagreement is itself the finding: it says the
apparent structure lives in the variance, not in the mean.

Implementation notes
--------------------

Written from first principles in numpy, consistent with how the Sharpe ratio
and maximum drawdown are computed in this repository rather than pulled from a
library. ``bashtage/arch`` provides ``arch.unitroot.VarianceRatio``, which was
read as a reference for the overlapping-estimator bias corrections; it is not
a dependency here.

The estimator uses **overlapping** :math:`q`-bar differences with Lo and
MacKinlay's finite-sample bias corrections. Non-overlapping differences would
be simpler but discard most of the sample — at :math:`q = 16` on 2,500 bars
there are only 156 non-overlapping windows, and the test would have almost no
power. Overlapping windows reuse the data at the cost of correlated terms,
which is what the :math:`m` denominator below corrects for.

Reference:
    Lo, A. W. and MacKinlay, A. C. (1988). "Stock Market Prices Do Not Follow
    Random Walks: Evidence from a Simple Specification Test." *Review of
    Financial Studies* 1(1), 41-66.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

#: Significance level for the random-walk null. Matches
#: :data:`statarb.cointegration.DEFAULT_SIGNIFICANCE`; fixed in code for the
#: same reason, that picking it after seeing the result is p-hacking.
DEFAULT_SIGNIFICANCE = 0.05

#: Horizons at which the test is run by default. Powers of two spanning a
#: half-life-scale range: a spread reverting over ~20 bars shows little at
#: q=2 and clearly at q=16, so testing one horizon alone risks missing the
#: structure entirely.
DEFAULT_HORIZONS = (2, 4, 8, 16)

#: Below this many observations the asymptotic distributions the z-statistics
#: rely on are not a usable approximation and the result would be a p-value
#: with no basis. Rejected loudly rather than returned as a number.
MIN_OBSERVATIONS = 32

#: Half-width of the band around 1.0 within which the point estimate is called
#: a random walk by :func:`interpret`. Only a plain-English reading of the
#: magnitude — the p-value, not this band, is what decides significance.
RANDOM_WALK_BAND = 0.05


@dataclass(frozen=True)
class VarianceRatioResult:
    """Variance ratio diagnostics for one series at one horizon.

    Attributes:
        horizon: The aggregation horizon :math:`q`, in bars.
        variance_ratio: The statistic itself. 1.0 under the random-walk null.
        z_homoskedastic: Standard normal z-statistic under Lo-MacKinlay's M1
            (homoskedastic IID increments).
        p_homoskedastic: Two-sided p-value for ``z_homoskedastic``.
        z_robust: Standard normal z-statistic under M2, which allows arbitrary
            conditional heteroskedasticity. **This is the one to read** on
            financial data, for the reason in the module docstring.
        p_robust: Two-sided p-value for ``z_robust``.
        n_obs: Number of observations in the underlying series.
    """

    horizon: int
    variance_ratio: float
    z_homoskedastic: float
    p_homoskedastic: float
    z_robust: float
    p_robust: float
    n_obs: int

    def rejects_random_walk(self, alpha: float = DEFAULT_SIGNIFICANCE) -> bool:
        """Whether the robust test rejects the random-walk null.

        Deliberately keyed to the robust p-value alone. Offering a choice of
        null here would invite picking whichever one rejects, and on data with
        kurtosis near 14 the homoskedastic version is the one that over-rejects
        — so the choice would only ever be exercised in the direction that
        manufactures a finding.
        """
        return bool(np.isfinite(self.p_robust) and self.p_robust < alpha)

    def interpretation(self) -> str:
        """Sign reading of the point estimate: reverting, trending, or neither."""
        return interpret(self.variance_ratio)

    def to_dict(self) -> dict:
        """Flat dict for CSV export."""
        return {
            "horizon": self.horizon,
            "variance_ratio": self.variance_ratio,
            "z_homoskedastic": self.z_homoskedastic,
            "p_homoskedastic": self.p_homoskedastic,
            "z_robust": self.z_robust,
            "p_robust": self.p_robust,
            "n_obs": self.n_obs,
            "rejects_random_walk": self.rejects_random_walk(),
            "interpretation": self.interpretation(),
        }


def interpret(vr: float, band: float = RANDOM_WALK_BAND) -> str:
    """Plain-English reading of a variance ratio point estimate.

    Describes the *magnitude* only. A ratio of 0.90 is mean-reverting in
    direction whether or not the departure from 1.0 is significant, so this
    must be read together with the p-value rather than instead of it — a
    strongly-worded interpretation attached to a p-value of 0.6 is noise with
    a label on it.
    """
    if not np.isfinite(vr):
        return "undefined"
    if vr < 1.0 - band:
        return "mean-reverting"
    if vr > 1.0 + band:
        return "trending"
    return "random walk"


def variance_ratio(series: np.ndarray, q: int) -> float:
    """The variance ratio statistic at horizon ``q``.

    Computes :math:`\\hat\\sigma_c^2(q) / \\hat\\sigma_a^2` using overlapping
    :math:`q`-bar differences, with Lo and MacKinlay's finite-sample bias
    corrections applied to both variance estimates.

    Two corrections are involved and both matter at the sample sizes here.
    The one-bar variance uses an ``nq - 1`` denominator because the mean is
    estimated. The :math:`q`-bar variance uses

    .. math:: m = q\\,(nq - q + 1)\\left(1 - \\frac{q}{nq}\\right)

    rather than the count of overlapping windows. That denominator makes the
    estimator unbiased under the null despite the overlap, and without it the
    statistic is biased *downward* — which would show up as spurious evidence
    of mean reversion, exactly the false positive this module is meant to rule
    out.

    Args:
        series: The series in **levels**, not differences. A spread or a log
            price, not returns; the function differences internally.
        q: Aggregation horizon in bars. Must be at least 2 — at ``q = 1`` the
            ratio is identically 1 by construction and carries no information.

    Returns:
        The variance ratio. Under a random walk this is approximately 1.0.

    Raises:
        ValueError: If ``q`` is below 2, the series is shorter than
            :data:`MIN_OBSERVATIONS`, the series is not longer than ``q``, or
            it contains non-finite values. Bad input is rejected rather than
            silently producing a number, because a variance ratio is plausible
            at any value and a wrong one would not look wrong.
    """
    series = np.asarray(series, dtype=float)
    if q < 2:
        raise ValueError(f"horizon q must be at least 2, got {q}")
    if series.size < MIN_OBSERVATIONS:
        raise ValueError(
            f"need at least {MIN_OBSERVATIONS} observations for the asymptotic "
            f"distribution to apply, got {series.size}"
        )
    if not np.all(np.isfinite(series)):
        raise ValueError("series must be finite")

    # Lo-MacKinlay notation: nq + 1 observations, hence nq one-bar increments.
    nq = series.size - 1
    if nq <= q:
        raise ValueError(
            f"series of {series.size} observations is too short for horizon "
            f"q={q}; need more than q increments"
        )

    # The drift estimate telescopes to the endpoints, which is the maximum
    # likelihood estimator under the null and is what the bias corrections
    # below assume was used.
    mu = (series[-1] - series[0]) / nq

    one_bar = np.diff(series)
    sigma_a_sq = float(np.sum((one_bar - mu) ** 2) / (nq - 1))
    if sigma_a_sq <= 0.0:
        # A constant series has no variance to take a ratio of. Returning nan
        # rather than raising keeps a degenerate column from aborting a sweep.
        return float("nan")

    q_bar = series[q:] - series[:-q]
    m = q * (nq - q + 1) * (1.0 - q / nq)
    sigma_c_sq = float(np.sum((q_bar - q * mu) ** 2) / m)

    return sigma_c_sq / sigma_a_sq


def _z_statistics(series: np.ndarray, q: int, vr: float) -> tuple[float, float]:
    """Homoskedastic and heteroskedasticity-robust z-statistics for ``vr``.

    Under the homoskedastic null (M1), :math:`\\sqrt{nq}\\,(VR(q) - 1)` is
    asymptotically normal with variance :math:`2(2q-1)(q-1)/(3q)`. That
    variance is a pure function of :math:`q` — it uses no information from the
    data beyond the sample size, which is precisely why it fails when the
    increments are heteroskedastic.

    Under the robust null (M2), the asymptotic variance is estimated from the
    data as :math:`\\theta = \\sum_{k=1}^{q-1} [2(q-k)/q]^2 \\delta_k`, where
    :math:`\\delta_k` is a heteroskedasticity-consistent estimator of the
    variance of the :math:`k`-th autocovariance. The triangular weights are
    the same ones that appear in the autocorrelation representation of
    :math:`VR(q)`, which is the sense in which M2 is the same statistic with
    its own variability measured rather than assumed.

    Returns:
        ``(z_homoskedastic, z_robust)``. Either is ``nan`` when its variance
        estimate degenerates, which propagates to a ``nan`` p-value rather
        than to a false rejection.
    """
    nq = series.size - 1
    one_bar = np.diff(series)
    mu = (series[-1] - series[0]) / nq
    centred = one_bar - mu

    phi_homo = 2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * nq)
    z_homo = (vr - 1.0) / np.sqrt(phi_homo) if phi_homo > 0 else float("nan")

    # Heteroskedasticity-consistent variance. Each delta_k is the ratio of the
    # sum of squared cross-products at lag k to the squared sum of squares,
    # so a period of high volatility inflates numerator and denominator
    # together and does not by itself register as autocorrelation.
    squared = centred**2
    denominator = float(np.sum(squared)) ** 2
    if denominator <= 0.0:
        return float(z_homo), float("nan")

    theta = 0.0
    for k in range(1, q):
        numerator = float(np.sum(squared[k:] * squared[:-k]))
        delta_k = numerator * nq / denominator
        weight = 2.0 * (q - k) / q
        theta += weight * weight * delta_k

    z_robust = (vr - 1.0) / np.sqrt(theta) if theta > 0 else float("nan")
    return float(z_homo), float(z_robust)


def _two_sided_p(z: float) -> float:
    """Two-sided normal p-value for a z-statistic.

    Two-sided rather than one-sided because both departures are meaningful:
    :math:`VR < 1` is mean reversion and :math:`VR > 1` is trending, and the
    null being tested is "random walk", not "not mean-reverting". A one-sided
    test would halve the p-value and would be the wrong test for the question.
    """
    if not np.isfinite(z):
        return float("nan")
    return float(2.0 * (1.0 - NormalDist().cdf(abs(z))))


def variance_ratio_test(
    series: np.ndarray,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> list[VarianceRatioResult]:
    """Run the variance ratio test at each horizon, under both nulls.

    Args:
        series: The series in **levels** — a spread or a log price, not
            returns. Passing returns would test whether the *returns* are a
            random walk, a different and generally uninteresting question.
        horizons: Aggregation horizons :math:`q`.

    Returns:
        One :class:`VarianceRatioResult` per horizon, in the order given.

    Note:
        The horizons are not independent tests — they are overlapping
        weightings of the same autocorrelation function, so their p-values are
        strongly positively correlated. Reading "rejected at one of four
        horizons" as four independent chances to reject would overstate the
        evidence, and the correct multiple-comparisons adjustment is not the
        Bonferroni one because of that dependence. Chow and Denning (1993)
        give a joint test based on the maximum absolute z-statistic across
        horizons; it is not implemented here, so the honest reading of these
        four rows is as one piece of evidence viewed at four resolutions
        rather than as four findings.
    """
    series = np.asarray(series, dtype=float)

    results: list[VarianceRatioResult] = []
    for q in horizons:
        vr = variance_ratio(series, q)
        if not np.isfinite(vr):
            z_homo = z_robust = float("nan")
        else:
            z_homo, z_robust = _z_statistics(series, q, vr)

        results.append(
            VarianceRatioResult(
                horizon=int(q),
                variance_ratio=float(vr),
                z_homoskedastic=z_homo,
                p_homoskedastic=_two_sided_p(z_homo),
                z_robust=z_robust,
                p_robust=_two_sided_p(z_robust),
                n_obs=int(series.size),
            )
        )
    return results
