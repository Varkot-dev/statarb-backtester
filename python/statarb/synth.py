"""Synthetic price generation with known ground-truth parameters.

Why synthetic data matters here: a backtester tested only on real prices can
only be checked for self-consistency. There is no external answer to compare
against, so a hedge ratio that is systematically 10% off, or a spread that is
subtly mis-centered, produces plausible output and goes unnoticed.

Generating series whose true parameters we chose lets us assert the estimator
recovers them. If rolling OLS on a series built with ``beta = 1.2`` does not
return approximately 1.2, the estimator is wrong — and we know that without
needing a market view.

All generation is seeded. Two runs with the same seed produce byte-identical
CSVs, which is what makes ``make backtest`` reproduce the README's numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CointegratedSpec:
    """Ground-truth parameters for a synthetic cointegrated pair.

    The construction is:

    .. code-block:: text

        c(t) = c(t-1) + N(0, sigma_common)           # common stochastic trend
        s(t) = phi * s(t-1) + N(0, innovation_sd)    # mean-reverting spread
        log P_A(t) = log a0 + beta * c(t) + s(t)
        log P_B(t) = log b0 + c(t)

    so that ``log P_A - beta * log P_B = log a0 + s(t)``, which is stationary
    by construction. The true hedge ratio is exactly ``beta`` and the true
    spread is ``s``.

    Both legs individually are I(1) — non-stationary random walks, as real
    prices are — which is the whole point. A pair of stationary series would
    be trivially tradeable and would not exercise the cointegration machinery.

    Attributes:
        n_bars: Number of observations to generate.
        beta: True hedge ratio (the cointegrating coefficient).
        half_life: Bars for the spread to revert halfway to its mean. This is
            the parameter that determines whether the strategy has anything to
            trade: too short and the spread never travels far enough to trip a
            2-sigma entry; too long and positions hit the holding cap before
            reverting.
        sigma_spread: Unconditional standard deviation of the spread.
        sigma_common: Per-bar volatility of the common trend.
        price_a0: Starting price of leg A.
        price_b0: Starting price of leg B.
    """

    n_bars: int
    beta: float
    half_life: float
    sigma_spread: float
    sigma_common: float
    price_a0: float = 100.0
    price_b0: float = 50.0

    def __post_init__(self) -> None:
        if self.n_bars < 2:
            raise ValueError(f"n_bars must be >= 2, got {self.n_bars}")
        if self.half_life <= 0:
            raise ValueError(f"half_life must be positive, got {self.half_life}")
        if self.sigma_spread <= 0:
            raise ValueError(f"sigma_spread must be positive, got {self.sigma_spread}")
        if self.sigma_common < 0:
            raise ValueError(f"sigma_common must be non-negative, got {self.sigma_common}")
        if self.price_a0 <= 0 or self.price_b0 <= 0:
            raise ValueError("starting prices must be positive")

    @property
    def phi(self) -> float:
        """AR(1) coefficient implied by the half-life.

        An AR(1) with coefficient ``phi`` decays by a factor ``phi`` per bar,
        so reaching half its starting deviation takes ``ln(0.5) / ln(phi)``
        bars. Inverting: ``phi = exp(-ln 2 / half_life)``.
        """
        return float(np.exp(-np.log(2.0) / self.half_life))

    @property
    def innovation_sd(self) -> float:
        """Innovation standard deviation giving the requested unconditional sd.

        A stationary AR(1) has unconditional variance
        ``sigma_e^2 / (1 - phi^2)``. Solving for ``sigma_e`` so the process has
        unconditional standard deviation ``sigma_spread``.
        """
        return float(self.sigma_spread * np.sqrt(1.0 - self.phi**2))


def business_dates(n: int, start: str = "2015-01-01") -> pd.DatetimeIndex:
    """Generate ``n`` consecutive business days starting at ``start``.

    Business days (not calendar days) because the annualization factor of 252
    assumes trading days. Using calendar days would make the reported
    annualized figures inconsistent with the stated convention.
    """
    return pd.bdate_range(start=start, periods=n)


def generate_cointegrated(
    spec: CointegratedSpec, seed: int, start_date: str = "2015-01-01"
) -> pd.DataFrame:
    """Generate a cointegrated pair from ``spec``.

    Args:
        spec: Ground-truth parameters.
        seed: PRNG seed. Uses ``numpy.random.default_rng``, whose stream is
            guaranteed stable across numpy versions (unlike the legacy
            ``RandomState``), so a regenerated dataset is byte-identical.
        start_date: First business date.

    Returns:
        DataFrame with columns ``date``, ``price_a``, ``price_b``, plus the
        ground-truth ``true_spread`` and ``common_trend`` columns for
        diagnostics. Only the first three are written to the engine's input
        file — the engine must never see the true spread, since knowing it
        would be the ultimate lookahead.
    """
    rng = np.random.default_rng(seed)
    n = spec.n_bars

    common_innovations = rng.normal(0.0, spec.sigma_common, size=n)
    common_trend = np.cumsum(common_innovations)

    # The spread is initialised at its stationary distribution rather than at
    # zero, so the series does not begin with an artificial burn-in during
    # which the spread is anomalously close to its mean.
    spread = np.empty(n)
    spread[0] = rng.normal(0.0, spec.sigma_spread)
    spread_innovations = rng.normal(0.0, spec.innovation_sd, size=n)
    phi = spec.phi
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + spread_innovations[t]

    log_a = np.log(spec.price_a0) + spec.beta * common_trend + spread
    log_b = np.log(spec.price_b0) + common_trend

    return pd.DataFrame(
        {
            "date": business_dates(n, start_date).strftime("%Y-%m-%d"),
            "price_a": np.exp(log_a),
            "price_b": np.exp(log_b),
            "true_spread": spread,
            "common_trend": common_trend,
        }
    )


def generate_independent(
    n_bars: int, seed: int, sigma: float = 0.015, start_date: str = "2015-01-01"
) -> pd.DataFrame:
    """Generate two independent random walks — a deliberately *non*-cointegrated pair.

    This is the negative control for the whole strategy. Two independent
    random walks have no stable relationship, so a pairs strategy run on them
    should not produce a meaningful edge. If it does, the backtester is
    manufacturing profit that is not there — which is the failure mode this
    entire repository is built to detect.

    The README reports the result on this pair alongside the cointegrated one
    precisely so a reader can see the strategy failing where it should fail.
    """
    rng = np.random.default_rng(seed)
    log_a = np.log(100.0) + np.cumsum(rng.normal(0.0, sigma, size=n_bars))
    log_b = np.log(50.0) + np.cumsum(rng.normal(0.0, sigma, size=n_bars))
    return pd.DataFrame(
        {
            "date": business_dates(n_bars, start_date).strftime("%Y-%m-%d"),
            "price_a": np.exp(log_a),
            "price_b": np.exp(log_b),
        }
    )
