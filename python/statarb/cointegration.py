"""Pair selection: correlation, cointegration, and why the difference matters.

Why correlation alone is insufficient for pairs trading
--------------------------------------------------------

This is the question interviewers ask, and the answer is not "correlation is
noisy" — it is a statement about the order of integration of the series.

Correlation measures whether two series move *together*. Cointegration measures
whether they stay *together*. These are different properties and neither
implies the other.

**Correlated but not cointegrated.** Take two independent random walks that
happen to drift upward over the sample. Their *returns* are uncorrelated, but
their *levels* can show correlation above 0.9 — this is the classic spurious
regression of Granger and Newbold (1974). The spread between them is itself a
random walk: it has no mean to revert to, and its variance grows without bound
as :math:`\\mathrm{Var}(s_t) = t\\sigma^2`. A pairs trade on such a spread has
no exit. The z-score keeps hitting 2, then 3, then 4, and the position keeps
losing. Every "mean reversion" the backtest sees in-sample is an artifact of
the finite window, and the strategy is really just a short-volatility bet that
blows up.

**Cointegrated but weakly correlated.** Two series can have a stable long-run
relationship while their short-term co-movement is modest — different
sensitivities to transient shocks, different microstructure. The spread is
stationary and tradeable even though a correlation screen would reject the pair.

Formally: :math:`X` and :math:`Y`, both I(1), are cointegrated if some linear
combination :math:`Y - \\beta X` is I(0) — stationary. That stationarity is
exactly the property a mean-reversion strategy needs, and it is what makes the
spread's mean and variance well-defined quantities to standardise against.
Correlation says nothing about the order of integration of the residual.

**The practical consequence.** Correlation is used here as a cheap pre-filter
to shrink the candidate set (it is O(n²) over the universe and fast), and
cointegration is the actual selection criterion. Ranking by correlation and
trading the top pairs is the single most common way a pairs-trading backtest
ends up trading spurious relationships.

Testing procedure
-----------------

Two tests are run:

- **Engle-Granger** (two-step): regress one leg on the other, then run an
  Augmented Dickey-Fuller test on the residual. Simple and directly matched to
  how the engine constructs its spread (OLS residual of log-price on
  log-price), which is why it is the primary criterion here — selecting under
  one model and trading under another is its own kind of inconsistency.
  Weakness: the result depends on which series is the regressand.

- **Johansen**: a maximum-likelihood test on the VECM representation that
  treats both series symmetrically and can identify multiple cointegrating
  relationships. Reported alongside as a robustness check.

A pair is accepted only if both agree, which is deliberately conservative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

#: Significance level for all hypothesis tests. Fixed in code rather than
#: tuned: choosing the threshold after seeing which pairs pass is p-hacking.
DEFAULT_SIGNIFICANCE = 0.05

#: Johansen trace-statistic critical value column indices, as returned by
#: statsmodels' ``coint_johansen``: columns are the 90%, 95%, and 99% critical
#: values respectively.
_JOHANSEN_CRIT_90, _JOHANSEN_CRIT_95, _JOHANSEN_CRIT_99 = 0, 1, 2


@dataclass(frozen=True)
class PairStats:
    """Diagnostics for one candidate pair.

    Attributes:
        name_a: Identifier of leg A.
        name_b: Identifier of leg B.
        correlation_levels: Pearson correlation of log *levels*. High values
            here are the trap described in the module docstring — this is the
            number that looks compelling and means little on its own.
        correlation_returns: Pearson correlation of log *returns*. More
            meaningful than the level correlation as a co-movement measure,
            because returns of an I(1) series are I(0) so the statistic is not
            spurious.
        hedge_ratio: Full-sample OLS slope of ``log_a`` on ``log_b``. Reported
            for research only; the engine re-estimates it on a rolling window,
            since a full-sample ratio would be lookahead if traded on.
        eg_pvalue: Engle-Granger cointegration test p-value.
        adf_pvalue: ADF p-value on the OLS residual (the spread).
        johansen_trace_stat: Johansen trace statistic for r=0.
        johansen_crit_95: 95% critical value for that statistic.
        half_life: Estimated mean-reversion half-life of the spread, in bars.
        spread_std: Standard deviation of the full-sample spread.
        n_obs: Number of observations used.
    """

    name_a: str
    name_b: str
    correlation_levels: float
    correlation_returns: float
    hedge_ratio: float
    eg_pvalue: float
    adf_pvalue: float
    johansen_trace_stat: float
    johansen_crit_95: float
    half_life: float
    spread_std: float
    n_obs: int

    def is_cointegrated(self, alpha: float = DEFAULT_SIGNIFICANCE) -> bool:
        """Whether both tests reject the null of no cointegration.

        Requiring agreement between two tests with different assumptions is
        conservative: it will reject some genuinely cointegrated pairs. That is
        the right trade-off here, because the cost of trading a spurious pair
        (an unbounded losing position) exceeds the cost of skipping a real one.
        """
        eg_ok = self.eg_pvalue < alpha
        johansen_ok = self.johansen_trace_stat > self.johansen_crit_95
        return bool(eg_ok and johansen_ok)

    def is_tradeable(
        self,
        alpha: float = DEFAULT_SIGNIFICANCE,
        min_half_life: float = 1.0,
        max_half_life: float = 120.0,
    ) -> bool:
        """Whether the pair is both cointegrated and has a usable half-life.

        Cointegration is necessary but not sufficient. A spread that reverts in
        under a bar cannot be traded on daily data — the signal is gone before
        the next-bar fill. A spread with a half-life of years is technically
        stationary but will not revert within any reasonable holding period,
        so positions hit the holding cap and exit on time rather than on
        signal.
        """
        return bool(
            self.is_cointegrated(alpha)
            and min_half_life <= self.half_life <= max_half_life
        )

    def to_dict(self) -> dict:
        """Flat dict for CSV export."""
        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "correlation_levels": self.correlation_levels,
            "correlation_returns": self.correlation_returns,
            "hedge_ratio": self.hedge_ratio,
            "eg_pvalue": self.eg_pvalue,
            "adf_pvalue": self.adf_pvalue,
            "johansen_trace_stat": self.johansen_trace_stat,
            "johansen_crit_95": self.johansen_crit_95,
            "half_life": self.half_life,
            "spread_std": self.spread_std,
            "n_obs": self.n_obs,
            "is_cointegrated": self.is_cointegrated(),
            "is_tradeable": self.is_tradeable(),
        }


def estimate_half_life(spread: np.ndarray) -> float:
    """Estimate the mean-reversion half-life of a spread, in bars.

    Fits the discrete Ornstein-Uhlenbeck regression

    .. math:: \\Delta s_t = \\alpha + \\lambda s_{t-1} + \\varepsilon_t

    A mean-reverting series has :math:`\\lambda < 0`: when the spread is above
    its mean, the next change is negative. The decay per bar is
    :math:`(1 + \\lambda)`, so the half-life is
    :math:`-\\ln 2 / \\ln(1 + \\lambda)`.

    .. warning::

       **This estimator is biased downward on near-unit-root series**, which
       matters for how its output should be read.

       :math:`s_{t-1}` is correlated with the innovation that produced it, so
       :math:`\\hat\\lambda` has a negative mean in finite samples even when the
       true :math:`\\lambda` is zero. In practice a genuine random walk usually
       returns a large but *finite* half-life (hundreds to thousands of bars)
       rather than ``inf``.

       This is the same bias that makes a naive OLS t-test on
       :math:`\\hat\\lambda` over-reject the unit-root null, and is why the
       Dickey-Fuller test uses its own non-standard critical values instead of
       the t-distribution.

       The consequence for this codebase: **do not treat a finite half-life as
       evidence of mean reversion.** Use it only as a tradeability filter, as
       :meth:`PairStats.is_tradeable` does — a half-life of 350 bars on daily
       data is 1.4 years and is rejected regardless of whether it is finite.
       Cointegration itself is established by ADF and Johansen, which have
       correct critical values; the half-life answers "on what timescale", not
       "is it stationary".

       The estimator is *consistent* — the error shrinks with sample size — as
       verified by ``test_half_life_estimator_is_consistent``.

    Returns:
        The half-life in bars, or ``inf`` when the fitted series is not
        mean-reverting at all (:math:`\\hat\\lambda \\geq 0`). Infinity
        propagates correctly through the tradeability band check, so no special
        case is needed at the call site.
    """
    spread = np.asarray(spread, dtype=float)
    if spread.size < 3:
        return float("inf")

    lagged = spread[:-1]
    delta = np.diff(spread)

    # OLS of delta on lagged with an intercept.
    design = np.column_stack([np.ones_like(lagged), lagged])
    try:
        coefficients, *_ = np.linalg.lstsq(design, delta, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")

    lam = float(coefficients[1])
    decay = 1.0 + lam
    if lam >= 0 or decay <= 0:
        # Not mean-reverting, or the estimate overshoots into oscillation.
        return float("inf")
    return float(-np.log(2.0) / np.log(decay))


def ols_hedge_ratio(log_a: np.ndarray, log_b: np.ndarray) -> tuple[float, float]:
    """Full-sample OLS of ``log_a`` on ``log_b``, returning ``(alpha, beta)``.

    Research use only. The engine re-estimates this on a trailing window;
    trading on a full-sample ratio would embed future information in every
    historical decision.
    """
    design = np.column_stack([np.ones_like(log_b), log_b])
    coefficients, *_ = np.linalg.lstsq(design, log_a, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def analyse_pair(
    prices_a: pd.Series,
    prices_b: pd.Series,
    name_a: str = "A",
    name_b: str = "B",
) -> PairStats:
    """Run the full diagnostic battery on one candidate pair.

    Args:
        prices_a: Price level series for leg A. Must be strictly positive.
        prices_b: Price level series for leg B, aligned to ``prices_a``.
        name_a: Identifier for leg A.
        name_b: Identifier for leg B.

    Returns:
        A populated :class:`PairStats`.

    Raises:
        ValueError: If the series are misaligned, too short, or contain
            non-positive prices. Non-positive prices are rejected rather than
            dropped because they indicate a data problem that silently
            shortening the sample would hide.
    """
    if len(prices_a) != len(prices_b):
        raise ValueError(
            f"series lengths differ: {name_a} has {len(prices_a)}, "
            f"{name_b} has {len(prices_b)}"
        )
    if len(prices_a) < 30:
        raise ValueError(
            f"need at least 30 observations for meaningful tests, got {len(prices_a)}"
        )

    a = np.asarray(prices_a, dtype=float)
    b = np.asarray(prices_b, dtype=float)
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("prices must be strictly positive (log prices are taken)")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("prices must be finite")

    log_a = np.log(a)
    log_b = np.log(b)

    correlation_levels = float(np.corrcoef(log_a, log_b)[0, 1])
    ret_a = np.diff(log_a)
    ret_b = np.diff(log_b)
    correlation_returns = float(np.corrcoef(ret_a, ret_b)[0, 1])

    alpha, beta = ols_hedge_ratio(log_a, log_b)
    spread = log_a - alpha - beta * log_b

    # Engle-Granger. statsmodels' `coint` runs the two-step procedure and
    # returns the p-value of the ADF test on the residual.
    _, eg_pvalue, _ = coint(log_a, log_b)

    # ADF directly on our own residual. This can differ slightly from the EG
    # p-value because `coint` uses its own regression and critical values;
    # reporting both makes the agreement (or lack of it) visible.
    adf_pvalue = float(adfuller(spread, autolag="AIC")[1])

    # Johansen trace test for r=0 (no cointegrating relationship).
    # det_order=0: a constant term in the cointegrating relationship, which is
    # right for log-price spreads that have a non-zero mean level.
    # k_ar_diff=1: one lagged difference in the VECM.
    data = np.column_stack([log_a, log_b])
    johansen = coint_johansen(data, det_order=0, k_ar_diff=1)
    trace_stat = float(johansen.lr1[0])
    crit_95 = float(johansen.cvt[0, _JOHANSEN_CRIT_95])

    return PairStats(
        name_a=name_a,
        name_b=name_b,
        correlation_levels=correlation_levels,
        correlation_returns=correlation_returns,
        hedge_ratio=beta,
        eg_pvalue=float(eg_pvalue),
        adf_pvalue=adf_pvalue,
        johansen_trace_stat=trace_stat,
        johansen_crit_95=crit_95,
        half_life=estimate_half_life(spread),
        spread_std=float(np.std(spread, ddof=1)),
        n_obs=len(a),
    )


def correlation_matrix(prices: pd.DataFrame, use_returns: bool = True) -> pd.DataFrame:
    """Correlation matrix across a universe of price series.

    Args:
        prices: DataFrame of price levels, one column per asset.
        use_returns: When True (the default), correlate log *returns*. When
            False, correlate log *levels*.

            The default is deliberate. Level correlations between I(1) series
            are spurious — two independent random walks routinely show |rho|
            above 0.9 over a few hundred bars — so a level-correlation screen
            selects for pairs that happen to have drifted together, which is
            precisely the wrong criterion. Return correlations are computed on
            I(0) series and are statistically meaningful.

    Returns:
        A symmetric correlation matrix.
    """
    if use_returns:
        data = np.log(prices).diff().dropna()
    else:
        data = np.log(prices)
    return data.corr()


def screen_universe(
    prices: pd.DataFrame,
    min_abs_correlation: float = 0.5,
    alpha: float = DEFAULT_SIGNIFICANCE,
) -> pd.DataFrame:
    """Screen every pair in a universe, ranked by cointegration evidence.

    The two-stage design mirrors real practice: a cheap correlation pre-filter
    reduces the O(n²) candidate set, then the expensive cointegration tests run
    only on survivors. The pre-filter is on *return* correlation for the reason
    given in :func:`correlation_matrix`.

    Note the multiple-comparisons problem this creates and that the README
    acknowledges: testing k pairs at alpha=0.05 yields roughly 0.05k false
    positives by construction. With a small universe this is manageable; at
    scale it requires a Bonferroni or FDR correction. This function does not
    apply one, which is a stated limitation rather than an oversight.

    Args:
        prices: DataFrame of price levels, one column per asset.
        min_abs_correlation: Return-correlation pre-filter threshold.
        alpha: Significance level for the cointegration tests.

    Returns:
        DataFrame of :class:`PairStats` rows, sorted by Engle-Granger p-value
        ascending (strongest evidence first).
    """
    columns = list(prices.columns)
    returns_corr = correlation_matrix(prices, use_returns=True)

    rows: list[dict] = []
    skipped = 0
    for i, name_a in enumerate(columns):
        for name_b in columns[i + 1 :]:
            if abs(returns_corr.loc[name_a, name_b]) < min_abs_correlation:
                skipped += 1
                continue
            try:
                stats = analyse_pair(
                    prices[name_a], prices[name_b], name_a=name_a, name_b=name_b
                )
            except ValueError:
                # A pair with bad data is skipped rather than aborting the whole
                # screen, but the count is reported so the omission is visible.
                skipped += 1
                continue
            row = stats.to_dict()
            row["is_cointegrated"] = stats.is_cointegrated(alpha)
            row["is_tradeable"] = stats.is_tradeable(alpha)
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "name_a",
                "name_b",
                "correlation_levels",
                "correlation_returns",
                "hedge_ratio",
                "eg_pvalue",
                "adf_pvalue",
                "johansen_trace_stat",
                "johansen_crit_95",
                "half_life",
                "spread_std",
                "n_obs",
                "is_cointegrated",
                "is_tradeable",
            ]
        )

    result = pd.DataFrame(rows).sort_values("eg_pvalue").reset_index(drop=True)
    result.attrs["n_skipped_by_prefilter"] = skipped
    return result
