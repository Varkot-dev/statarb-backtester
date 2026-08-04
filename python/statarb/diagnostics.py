"""Why a pairs strategy underperforms: mechanism, not excuses.

A backtest that reports a weak result leaves the reader with no way to tell
which of three very different things happened:

1. the engine is broken;
2. the strategy is mis-specified for this data;
3. there is genuinely no edge.

These are not interchangeable, and "the strategy didn't work" is a
non-explanation. This module distinguishes them with controlled experiments on
data whose ground truth is known.

The headline finding
--------------------

**The ratio of the z-score window to the spread's half-life governs whether the
signal works at all**, and it dominates signal-to-noise.

Measured on synthetic data at *constant* signal-to-noise, varying only the
half-life:

.. code-block:: text

    half-life   Sharpe    win rate    trades
        5       +1.41      84.5%        58
       10       +0.69      63.5%        52
       15       +0.08      60.4%        48
       20       -0.02      61.5%        52
       30       -0.23      59.2%        49

Trade *count* barely moves. The win *rate* collapses. So the problem is not
fewer opportunities — it is worse ones.

The mechanism
-------------

The z-score standardises the spread against a trailing window. When that window
is only a small multiple of the half-life, the window is dominated by the very
deviation being measured: the spread sits elevated for roughly a half-life, and
the rolling mean drifts up with it. The z-score therefore reads "normal"
precisely when the spread is most extreme.

The estimator defeats itself. Formally, for an AR(1) spread with coefficient
:math:`\\phi = 2^{-1/h}`, the trailing mean over :math:`w` bars has expectation
that tracks a geometrically-weighted average of recent deviations; the
contamination is :math:`O(\\phi^w)` and only becomes negligible once
:math:`w \\gg h`.

Fixing the half-life at 20 and sweeping only the window confirms it:

.. code-block:: text

    window   w/h    Sharpe   win rate   trading PnL
        40   2.0    +0.00     48.5%        -$544
        60   3.0    -0.02     61.5%        -$270
        90   4.5    +0.38     70.3%     +$10,954
       120   6.0    +0.58     75.8%     +$18,741
       250  12.5    +1.04     90.9%     +$26,770

Same data, same strategy, same costs. Only the window changed.

Why this does *not* mean alpha was found
-----------------------------------------

Applying this fix per-pair to real market data flips every Sharpe positive, two
with :math:`p < 0.05`. That result does not survive scrutiny, and the reason is
the most important thing in this module.

Choosing a window per pair **is a specification search**. The Deflated Sharpe
Ratio of Bailey & López de Prado (2014) gives the Sharpe one should expect from
the best of :math:`N` trials under the null of no skill. On 2,515 bars with
~30 configurations explored, that expectation is **+0.656** — against a best
observed value of **+0.654**. Indistinguishable.

The honest test is walk-forward: estimate the half-life on the first 60% of the
sample, set the window from it, and evaluate on the untouched remainder. That
gives a mean out-of-sample Sharpe of **-0.029**.

So: the methodological finding is real and reproducible. The alpha is not.
That distinction is the entire point.

See :func:`half_life_sensitivity`, :func:`window_ratio_sensitivity`,
:func:`deflated_sharpe_threshold`, and :func:`walk_forward` — each returns the
data behind one of the tables above.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .cointegration import estimate_half_life, ols_hedge_ratio
from .io import read_bars, read_metrics, read_trades, write_prices
from .synth import CointegratedSpec, generate_cointegrated

#: Euler-Mascheroni constant, used in the expected-maximum formula.
EULER_MASCHERONI = 0.5772156649015329

#: Minimum window-to-half-life ratio for the z-score to be usable.
#:
#: Below this the trailing mean is contaminated by the deviation being measured
#: (see the module docstring). Five is where the measured curves turn clearly
#: positive; it is a rule of thumb from these experiments, not a theorem.
MIN_WINDOW_HALF_LIFE_RATIO = 5.0


def _run_engine(
    engine: Path, prices_csv: Path, out_dir: Path, **flags: object
) -> dict[str, float] | None:
    """Invoke the engine, returning its metrics or ``None`` if it refused.

    A refusal is a legitimate outcome here — many configurations in these sweeps
    are invalid (a window longer than the sample, for instance) — so it is
    reported as ``None`` rather than raised.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(engine), "backtest", "--prices", str(prices_csv),
           "--out-dir", str(out_dir), "--quiet"]
    for key, value in flags.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return read_metrics(out_dir / "metrics.csv")


@dataclass(frozen=True)
class SensitivityRow:
    """One point on a sensitivity curve."""

    half_life: float
    zscore_window: int
    window_ratio: float
    sharpe: float
    win_rate: float
    n_trades: int
    trading_pnl: float

    def to_dict(self) -> dict:
        return {
            "half_life": self.half_life,
            "zscore_window": self.zscore_window,
            "window_ratio": self.window_ratio,
            "sharpe": self.sharpe,
            "win_rate": self.win_rate,
            "n_trades": self.n_trades,
            "trading_pnl": self.trading_pnl,
        }


def half_life_sensitivity(
    engine: Path,
    workdir: Path,
    half_lives: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 30.0),
    sigma_spread: float = 0.05,
    sigma_common: float = 0.005,
    zscore_window: int = 60,
    seed: int = 7,
    n_bars: int = 2520,
) -> list[SensitivityRow]:
    """Vary the half-life at constant signal-to-noise, window held fixed.

    This isolates the effect of reversion *speed*. Signal size, noise, costs,
    thresholds, and the window are all constant across rows, so any change in
    Sharpe is attributable to the half-life alone.
    """
    rows: list[SensitivityRow] = []
    for half_life in half_lives:
        spec = CointegratedSpec(
            n_bars=n_bars, beta=1.2, half_life=half_life,
            sigma_spread=sigma_spread, sigma_common=sigma_common,
        )
        prices = write_prices(
            generate_cointegrated(spec, seed=seed), workdir / "hl.csv"
        )
        metrics = _run_engine(
            engine, prices, workdir / "out", zscore_window=zscore_window,
            hedge_window=zscore_window,
        )
        if metrics is None:
            continue
        rows.append(
            SensitivityRow(
                half_life=half_life,
                zscore_window=zscore_window,
                window_ratio=zscore_window / half_life,
                sharpe=metrics["sharpe_ratio"],
                win_rate=metrics["win_rate"],
                n_trades=int(metrics["n_trades"]),
                trading_pnl=metrics["net_pnl"] - metrics["total_interest"],
            )
        )
    return rows


def window_ratio_sensitivity(
    engine: Path,
    workdir: Path,
    half_life: float = 20.0,
    ratios: tuple[float, ...] = (2.0, 3.0, 4.5, 6.0, 9.0, 12.5),
    sigma_spread: float = 0.05,
    sigma_common: float = 0.005,
    seed: int = 7,
    n_bars: int = 2520,
) -> list[SensitivityRow]:
    """Hold the data fixed and vary only the window.

    The cleanest demonstration of the finding: identical prices, identical
    strategy, identical costs. Only the estimator's window changes.
    """
    spec = CointegratedSpec(
        n_bars=n_bars, beta=1.2, half_life=half_life,
        sigma_spread=sigma_spread, sigma_common=sigma_common,
    )
    prices = write_prices(
        generate_cointegrated(spec, seed=seed), workdir / "ratio.csv"
    )
    rows: list[SensitivityRow] = []
    for ratio in ratios:
        window = int(round(half_life * ratio))
        metrics = _run_engine(
            engine, prices, workdir / "out", zscore_window=window,
            hedge_window=window,
        )
        if metrics is None:
            continue
        rows.append(
            SensitivityRow(
                half_life=half_life,
                zscore_window=window,
                window_ratio=ratio,
                sharpe=metrics["sharpe_ratio"],
                win_rate=metrics["win_rate"],
                n_trades=int(metrics["n_trades"]),
                trading_pnl=metrics["net_pnl"] - metrics["total_interest"],
            )
        )
    return rows


def expected_max_sharpe_under_null(n_trials: int, n_observations: int,
                                   bars_per_year: float = 252.0) -> float:
    """Expected best Sharpe from ``n_trials`` independent strategies with no skill.

    This is the core of the Deflated Sharpe Ratio (Bailey & López de Prado,
    2014). Under the null, each trial's Sharpe estimate is approximately normal
    with standard error :math:`\\sqrt{1/T}` (annualized by
    :math:`\\sqrt{\\text{bars per year}}`). The expected maximum of ``N`` such
    draws is

    .. math::

        E[\\max] \\approx \\sigma\\left[(1-\\gamma)\\,\\Phi^{-1}(1 - 1/N)
                                  + \\gamma\\,\\Phi^{-1}(1 - 1/(Ne))\\right]

    with :math:`\\gamma` the Euler-Mascheroni constant.

    **Why this matters more than any single p-value.** Searching over
    configurations and reporting the best one, with an unadjusted p-value, is
    the single most common way a backtest overstates its result. This function
    gives the number that best result must beat to mean anything.

    Args:
        n_trials: Number of configurations actually tried. Be honest — this
            includes every window, threshold, and pair combination examined,
            not just the ones written down.
        n_observations: Length of the return series.
        bars_per_year: Annualization factor.

    Returns:
        The annualized Sharpe that a no-skill search of this size is expected
        to produce as its best result.
    """
    if n_trials < 2:
        return 0.0
    normal = NormalDist()
    standard_error = np.sqrt(1.0 / n_observations) * np.sqrt(bars_per_year)
    return float(
        standard_error
        * (
            (1.0 - EULER_MASCHERONI) * normal.inv_cdf(1.0 - 1.0 / n_trials)
            + EULER_MASCHERONI * normal.inv_cdf(1.0 - 1.0 / (n_trials * np.e))
        )
    )


def sharpe_standard_error(
    sharpe_per_bar: float, n_observations: int, skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Standard error of a Sharpe estimate, corrected for non-normal returns.

    .. math::

        \\mathrm{SE}(\\widehat{SR}) = \\sqrt{
            \\frac{1 - \\gamma_3 SR + \\frac{\\gamma_4 - 1}{4} SR^2}{T - 1}}

    where :math:`\\gamma_3` is skewness and :math:`\\gamma_4` is (non-excess)
    kurtosis. Setting them to 0 and 3 recovers the Gaussian :math:`\\sqrt{1/T}`.

    **Why the correction matters for this strategy specifically.** Pairs-trading
    returns are strongly non-normal: many small mean-reversion wins against
    occasional large stop-loss losses. The measured NAV returns here have
    kurtosis ≈ 14 against a normal value of 3. Ignoring that understates the
    standard error, which overstates significance — in the direction that
    flatters the result.

    All arguments must be in **per-bar** units. This is the unit convention that
    the reference implementation surveyed below gets wrong, so it is enforced by
    the parameter name rather than left to a docstring.

    Reference: Mertens (2002); Lo (2002), *The Statistics of Sharpe Ratios*.
    The form here follows Bailey & López de Prado (2014).

    Prior art surveyed before writing this: ``mnemox-ai/deflated-sharpe``
    (Apache-2.0) implements the same correction, and reading it is what prompted
    adding skew/kurtosis here rather than keeping the Gaussian form. Its
    ``deflated_sharpe_ratio`` was **not** adopted: it documents ``observed_sr``
    as annualized but compares it against an un-annualized threshold and
    standard error, inflating the statistic by :math:`\\sqrt{252} \\approx 15.9`
    (feeding it a realistic Sharpe of 0.65 returns a 28-sigma result). The idea
    was worth taking; the implementation was not.
    """
    if n_observations < 2:
        return float("inf")
    variance = (
        1.0
        - skewness * sharpe_per_bar
        + ((kurtosis - 1.0) / 4.0) * sharpe_per_bar**2
    ) / (n_observations - 1)
    return float(np.sqrt(max(variance, 1e-12)))


def deflated_sharpe_threshold(
    observed_sharpe: float, n_trials: int, n_observations: int,
    bars_per_year: float = 252.0, skewness: float = 0.0, kurtosis: float = 3.0,
) -> dict[str, float | bool]:
    """Compare an observed Sharpe against what the search alone would produce.

    Returns the observed value, the no-skill expectation for a search of that
    size, the deflated statistic and its p-value, and whether the observation
    clears its own threshold. **A result that does not clear the threshold its
    own search created is not evidence of skill, whatever its nominal p-value
    says.**

    Args:
        observed_sharpe: The best **annualized** Sharpe found by the search.
        n_trials: How many configurations were actually tried. Count honestly:
            every window, threshold, and pair combination examined, not only
            the ones written down.
        n_observations: Length of the return series.
        bars_per_year: Annualization factor.
        skewness: Skewness of the per-bar returns.
        kurtosis: Non-excess kurtosis of the per-bar returns (3 = normal).

    All internal arithmetic is done in per-bar units and converted back at the
    end, because mixing annualized and per-bar quantities is the single easiest
    way to get this wrong by a factor of :math:`\\sqrt{252}`.
    """
    annualization = np.sqrt(bars_per_year)
    threshold_annual = expected_max_sharpe_under_null(
        n_trials, n_observations, bars_per_year
    )

    observed_per_bar = observed_sharpe / annualization
    threshold_per_bar = threshold_annual / annualization
    standard_error = sharpe_standard_error(
        observed_per_bar, n_observations, skewness, kurtosis
    )

    deflated = (observed_per_bar - threshold_per_bar) / standard_error
    p_value = 1.0 - NormalDist().cdf(deflated)

    return {
        "observed_sharpe": observed_sharpe,
        "n_trials": n_trials,
        "expected_max_under_null": threshold_annual,
        "excess_over_null": observed_sharpe - threshold_annual,
        "deflated_sharpe_statistic": float(deflated),
        "p_value": float(p_value),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "survives": bool(observed_sharpe > threshold_annual),
    }


@dataclass(frozen=True)
class WalkForwardRow:
    """One pair's out-of-sample result under walk-forward selection."""

    pair: str
    in_sample_half_life: float
    selected_window: int
    oos_sharpe: float
    oos_n_trades: int
    oos_t_statistic: float

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "in_sample_half_life": self.in_sample_half_life,
            "selected_window": self.selected_window,
            "oos_sharpe": self.oos_sharpe,
            "oos_n_trades": self.oos_n_trades,
            "oos_t_statistic": self.oos_t_statistic,
        }


def walk_forward(
    engine: Path,
    workdir: Path,
    prices: pd.DataFrame,
    pair_name: str,
    split_fraction: float = 0.6,
    ratio: float = MIN_WINDOW_HALF_LIFE_RATIO,
) -> WalkForwardRow | None:
    """Select the window in-sample, evaluate out-of-sample.

    This is the only test of the window rule that is not circular. The
    half-life is estimated on the first ``split_fraction`` of the sample and the
    window is derived from it; the backtest then runs on the remainder, which
    played no part in the choice.

    Anything less than this — including "I picked the window that made the
    result look right" — measures the analyst, not the strategy.

    Returns ``None`` when the out-of-sample segment is too short to support the
    selected window, which is itself informative: a long half-life demands a
    long window and therefore a long sample, and most real pairs do not have
    one.
    """
    split = int(len(prices) * split_fraction)
    in_sample = prices.iloc[:split]
    out_of_sample = prices.iloc[split:].reset_index(drop=True)

    log_a = np.log(in_sample["price_a"].to_numpy(dtype=float))
    log_b = np.log(in_sample["price_b"].to_numpy(dtype=float))
    _, beta = ols_hedge_ratio(log_a, log_b)
    half_life = estimate_half_life(log_a - beta * log_b)
    if not np.isfinite(half_life):
        half_life = 200.0

    window = int(round(half_life * ratio))
    # The window must leave enough post-warm-up bars to trade. A quarter of the
    # out-of-sample segment is a generous cap; beyond it there is no test left.
    if window > len(out_of_sample) // 4:
        return None

    prices_csv = write_prices(out_of_sample, workdir / "wf.csv")
    metrics = _run_engine(
        engine, prices_csv, workdir / "wf_out", zscore_window=window,
        hedge_window=window,
    )
    if metrics is None:
        return None

    trades = read_trades(workdir / "wf_out" / "trades.csv")
    if len(trades) > 1:
        pnl = trades["pnl_net"].to_numpy(dtype=float)
        t_stat = float(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl))))
    else:
        t_stat = float("nan")

    return WalkForwardRow(
        pair=pair_name,
        in_sample_half_life=float(half_life),
        selected_window=window,
        oos_sharpe=metrics["sharpe_ratio"],
        oos_n_trades=int(metrics["n_trades"]),
        oos_t_statistic=t_stat,
    )
