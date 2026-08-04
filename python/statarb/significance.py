"""Statistical significance of a backtest result.

Why this module exists
----------------------

A backtest reports a single number — say a Sharpe of 0.74 — computed from one
sample path. The number that actually matters is whether that result is
distinguishable from luck, and a point estimate cannot answer that.

The concrete case in this repository: the negative-control dataset (two
independent random walks, no relationship to trade) produced a total return of
+9.2% from trading over ten years. Read as a point estimate that looks like an
edge. Its t-statistic is 0.86 with p = 0.39 — entirely consistent with zero.
Fifty-three trades with a per-trade standard deviation of $1,465 have a
standard error of about $200 on the mean, so several thousand dollars of
cumulative PnL is well within noise.

Reporting the point estimate alone would have made a no-edge dataset look
mildly profitable. That is the specific way a backtest misleads even when the
arithmetic behind it is correct, and it is why the README leads with the
t-statistic rather than with total return.

What is computed
----------------

- **Per-trade t-statistic**: is mean trade PnL distinguishable from zero?
- **Sharpe standard error**: Lo (2002) gives
  :math:`\\mathrm{SE}(\\hat{SR}) \\approx \\sqrt{(1 + \\hat{SR}^2/2)/n}` for IID
  returns, which converts a Sharpe into a confidence interval.
- **Stationary bootstrap** (Politis & Romano, 1994): resamples the return
  series in blocks of random length, preserving the serial dependence that
  pairs-trading returns exhibit. The IID formulas above understate uncertainty
  precisely because those returns are autocorrelated within a held position, so
  the bootstrap interval is the more honest one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Bootstrap resamples. 2000 is enough for stable 95% interval endpoints
#: without making the pipeline slow.
DEFAULT_N_BOOTSTRAP = 2000

#: Expected block length for the stationary bootstrap, in bars. Should exceed
#: the typical holding period so that a resampled block contains whole trades
#: rather than fragments; the average holding period here is ~10-15 bars.
DEFAULT_BLOCK_LENGTH = 20.0


@dataclass(frozen=True)
class SignificanceResult:
    """Significance diagnostics for one backtest.

    Attributes:
        n_trades: Number of closed trades.
        mean_trade_pnl: Mean net PnL per trade.
        se_trade_pnl: Standard error of that mean.
        t_statistic: ``mean / se``. Under the null of no edge this is
            approximately t-distributed with ``n_trades - 1`` degrees of
            freedom.
        p_value: Two-sided p-value for the null of zero mean trade PnL.
        sharpe: The point-estimate annualized Sharpe ratio.
        sharpe_se: Lo (2002) IID standard error of the Sharpe.
        sharpe_ci_low: Lower bound of the 95% bootstrap interval.
        sharpe_ci_high: Upper bound of the 95% bootstrap interval.
        bootstrap_p_value: Fraction of bootstrap resamples with Sharpe <= 0.
            A one-sided test of "the strategy has no edge".
        is_significant: Whether the per-trade t-test rejects at 5%.
    """

    n_trades: int
    mean_trade_pnl: float
    se_trade_pnl: float
    t_statistic: float
    p_value: float
    sharpe: float
    sharpe_se: float
    sharpe_ci_low: float
    sharpe_ci_high: float
    bootstrap_p_value: float
    is_significant: bool

    def to_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "mean_trade_pnl": self.mean_trade_pnl,
            "se_trade_pnl": self.se_trade_pnl,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "sharpe": self.sharpe,
            "sharpe_se": self.sharpe_se,
            "sharpe_ci_low": self.sharpe_ci_low,
            "sharpe_ci_high": self.sharpe_ci_high,
            "bootstrap_p_value": self.bootstrap_p_value,
            "is_significant": self.is_significant,
        }

    def verdict(self) -> str:
        """A one-line plain-English reading, for the README."""
        if self.n_trades < 2:
            return "too few trades to assess significance"
        if self.is_significant:
            return (
                f"mean trade PnL is distinguishable from zero "
                f"(t={self.t_statistic:.2f}, p={self.p_value:.3f})"
            )
        return (
            f"mean trade PnL is NOT distinguishable from zero "
            f"(t={self.t_statistic:.2f}, p={self.p_value:.3f}) — consistent with no edge"
        )


def _student_t_sf(t: float, df: int) -> float:
    """Two-sided survival function of the t-distribution.

    Implemented via the regularised incomplete beta function using only numpy,
    to keep ``scipy`` out of the dependency list for one function. The identity
    is the standard one:

    .. math:: P(|T| > t) = I_{df/(df + t^2)}(df/2, 1/2)
    """
    if df <= 0:
        return float("nan")
    t = abs(float(t))
    if not np.isfinite(t):
        return 0.0
    x = df / (df + t * t)
    return float(_betainc(df / 2.0, 0.5, x))


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b), via continued fraction.

    Numerical Recipes' ``betacf`` algorithm. Accurate to ~1e-12 over the range
    used here, which is far beyond what a reported p-value needs.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Log of the leading factor, computed in logs to avoid overflow for large a.
    from math import exp, lgamma, log

    front = exp(
        lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x)
    )

    # The continued fraction converges quickly for x < (a+1)/(a+b+2); otherwise
    # use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a).
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)

    tiny = 1e-30
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d

    for m in range(1, 300):
        m2 = 2 * m
        # Even step.
        numerator = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        # Odd step.
        numerator = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break

    return front * h / a


def stationary_bootstrap_sharpe(
    returns: np.ndarray,
    bars_per_year: float,
    risk_free_annual: float,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    block_length: float = DEFAULT_BLOCK_LENGTH,
    seed: int = 12345,
) -> np.ndarray:
    """Bootstrap the Sharpe ratio, preserving serial dependence.

    Uses the stationary bootstrap of Politis & Romano (1994): starting from a
    random index, copy consecutive observations, and at each step restart from
    a new random index with probability ``1 / block_length``. Block lengths are
    therefore geometrically distributed with mean ``block_length``.

    A plain IID bootstrap would destroy the autocorrelation that pairs-trading
    returns have — a position held for fifteen bars produces fifteen related
    observations — and would therefore report a confidence interval that is too
    narrow. Blocks preserve that structure.

    Args:
        returns: Per-bar simple returns.
        bars_per_year: Annualization factor.
        risk_free_annual: Annual risk-free rate.
        n_bootstrap: Number of resamples.
        block_length: Expected block length in bars.
        seed: PRNG seed, so the reported interval is reproducible.

    Returns:
        Array of ``n_bootstrap`` bootstrap Sharpe ratios.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n < 2:
        return np.array([])

    rng = np.random.default_rng(seed)
    rf_bar = (1.0 + risk_free_annual) ** (1.0 / bars_per_year) - 1.0
    restart_prob = 1.0 / max(1.0, block_length)
    annualization = np.sqrt(bars_per_year)

    out = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        # Vectorised stationary bootstrap: draw all restart decisions and all
        # restart positions up front, then build the index path cumulatively.
        starts = rng.integers(0, n, size=n)
        restarts = rng.random(n) < restart_prob
        restarts[0] = True

        indices = np.empty(n, dtype=np.int64)
        current = starts[0]
        for i in range(n):
            if restarts[i]:
                current = starts[i]
            else:
                current = (current + 1) % n
            indices[i] = current

        sample = returns[indices]
        excess = sample - rf_bar
        sd = excess.std(ddof=1)
        out[b] = 0.0 if sd < 1e-12 else excess.mean() / sd * annualization

    return out


def assess(
    trades: pd.DataFrame,
    navs: np.ndarray,
    bars_per_year: float = 252.0,
    risk_free_annual: float = 0.04,
    alpha: float = 0.05,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 12345,
) -> SignificanceResult:
    """Assess whether a backtest result is distinguishable from luck.

    Args:
        trades: The engine's trade log; must have a ``pnl_net`` column.
        navs: The NAV series, used for the Sharpe bootstrap.
        bars_per_year: Annualization factor.
        risk_free_annual: Annual risk-free rate.
        alpha: Significance level for the per-trade t-test.
        n_bootstrap: Bootstrap resamples.
        seed: PRNG seed.

    Returns:
        A populated :class:`SignificanceResult`.
    """
    navs = np.asarray(navs, dtype=float)

    n_trades = len(trades)
    if n_trades >= 2:
        pnl = trades["pnl_net"].to_numpy(dtype=float)
        mean_pnl = float(pnl.mean())
        se_pnl = float(pnl.std(ddof=1) / np.sqrt(n_trades))
        t_stat = mean_pnl / se_pnl if se_pnl > 0 else 0.0
        p_value = _student_t_sf(t_stat, n_trades - 1)
    else:
        mean_pnl = se_pnl = t_stat = 0.0
        p_value = float("nan")

    # Sharpe point estimate and its IID standard error.
    if len(navs) >= 3:
        returns = navs[1:] / navs[:-1] - 1.0
        rf_bar = (1.0 + risk_free_annual) ** (1.0 / bars_per_year) - 1.0
        excess = returns - rf_bar
        sd = excess.std(ddof=1)
        sharpe_per_bar = 0.0 if sd < 1e-12 else float(excess.mean() / sd)
        sharpe = sharpe_per_bar * float(np.sqrt(bars_per_year))
        # Lo (2002), IID case.
        #
        # UNIT DISCIPLINE. This line was wrong for a while, in the same way and
        # by the same factor that CREDITS.md criticises another project for:
        # `sharpe` above is ANNUALIZED, but Lo's SE = sqrt((1 + SR^2/2)/T) is
        # unit-consistent only when SR and T share a time base. Computing it
        # from the annualized Sharpe against a bar count understated the
        # standard error by sqrt(252) ~ 16x, and a nominal 95% interval
        # achieved measured coverage of 0.095.
        #
        # The tell was visible in the shipped output the whole time: this SE
        # said 0.0225 while the bootstrap CI half-width on the same series said
        # 0.5725 — two uncertainty measures in one dataclass disagreeing 25x.
        #
        # So: compute in per-bar units, annualize once, at the end. The
        # intermediate is named `_per_bar` so a future edit cannot re-introduce
        # the ambiguity by reading a bare `sharpe` and assuming a time base.
        sharpe_se_per_bar = float(
            np.sqrt((1.0 + 0.5 * sharpe_per_bar**2) / len(returns))
        )
        sharpe_se = sharpe_se_per_bar * float(np.sqrt(bars_per_year))

        boot = stationary_bootstrap_sharpe(
            returns,
            bars_per_year=bars_per_year,
            risk_free_annual=risk_free_annual,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        if len(boot) > 0:
            ci_low, ci_high = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
            bootstrap_p = float(np.mean(boot <= 0.0))
        else:
            ci_low = ci_high = bootstrap_p = float("nan")
    else:
        sharpe = sharpe_se = 0.0
        ci_low = ci_high = bootstrap_p = float("nan")

    return SignificanceResult(
        n_trades=n_trades,
        mean_trade_pnl=mean_pnl,
        se_trade_pnl=se_pnl,
        t_statistic=float(t_stat),
        p_value=float(p_value),
        sharpe=sharpe,
        sharpe_se=sharpe_se,
        sharpe_ci_low=ci_low,
        sharpe_ci_high=ci_high,
        bootstrap_p_value=bootstrap_p,
        is_significant=bool(np.isfinite(p_value) and p_value < alpha),
    )
