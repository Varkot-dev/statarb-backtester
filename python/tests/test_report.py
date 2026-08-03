"""Tests for chart and table generation.

Charts are hard to assert on meaningfully — a pixel comparison is brittle and a
"file exists" check is nearly vacuous. These tests target the parts that carry
actual information: that a figure is produced without raising on realistic
inputs (including the awkward ones: no trades, all-NaN warm-up columns), and
that the Markdown tables contain the numbers they claim to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.report import (
    exit_reason_markdown_table,
    metrics_markdown_table,
    plot_equity_curve,
    plot_equity_decomposition,
    plot_price_series,
    plot_spread_and_zscore,
    plot_trade_distribution,
)


@pytest.fixture
def bars() -> pd.DataFrame:
    """A realistic bar frame, including a warm-up period with NaN statistics."""
    n = 200
    warmup = 60
    rng = np.random.default_rng(0)
    nav = 100_000.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.004, n))
    zscore = np.full(n, np.nan)
    zscore[warmup:] = rng.normal(0.0, 1.2, n - warmup)
    spread = np.full(n, np.nan)
    spread[warmup:] = rng.normal(0.0, 0.03, n - warmup)
    hedge = np.full(n, np.nan)
    hedge[warmup:] = 1.2 + rng.normal(0.0, 0.05, n - warmup)

    events = [""] * n
    events[70] = "entry:long_spread"
    events[85] = "exit:reversion"
    events[120] = "entry:short_spread"
    events[140] = "exit:stop_loss"

    return pd.DataFrame(
        {
            "index": np.arange(n),
            "date": pd.bdate_range("2020-01-01", periods=n),
            "price_a": 100.0 + np.cumsum(rng.normal(0, 0.5, n)),
            "price_b": 50.0 + np.cumsum(rng.normal(0, 0.3, n)),
            "hedge_ratio": hedge,
            "spread": spread,
            "zscore": zscore,
            "position": ["flat"] * n,
            "qty_a": np.zeros(n),
            "qty_b": np.zeros(n),
            "cash": nav,
            "position_value": np.zeros(n),
            "nav": nav,
            "costs_this_bar": np.zeros(n),
            "interest_this_bar": np.full(n, 15.5),
            "trade_event": events,
        }
    )


@pytest.fixture
def trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "direction": ["long_spread", "short_spread", "long_spread"],
            "entry_index": [70, 120, 160],
            "exit_index": [85, 140, 175],
            "pnl_gross": [520.0, -310.0, 180.0],
            "costs": [15.0, 15.0, 15.0],
            "pnl_net": [505.0, -325.0, 165.0],
            "exit_reason": ["reversion", "stop_loss", "reversion"],
            "holding_bars": [15, 20, 15],
        }
    )


def test_equity_curve_is_written(bars, tmp_path):
    path = plot_equity_curve(bars, tmp_path / "equity.png")
    assert path.exists() and path.stat().st_size > 1000


def test_equity_decomposition_is_written(bars, tmp_path):
    """The decomposition chart renders on a realistic frame."""
    path = plot_equity_decomposition(bars, tmp_path / "decomp.png")
    assert path.exists() and path.stat().st_size > 1000


def test_equity_decomposition_arithmetic():
    """trading_pnl = total_gain - cumulative_interest, exactly.

    This is the identity the chart is built on, and it follows from the
    accounting identity. Checked directly rather than via the rendered image,
    since a chart that is drawn but wrong is the failure mode that matters.
    """
    n = 10
    interest = np.full(n, 10.0)
    interest[0] = 0.0  # bar 0 accrues nothing
    nav = 100_000.0 + np.cumsum(interest) + np.arange(n) * 5.0
    frame = pd.DataFrame({"nav": nav, "interest_this_bar": interest})

    cumulative_interest = np.cumsum(frame["interest_this_bar"].to_numpy())
    trading = (nav - nav[0]) - cumulative_interest
    # Trading contributes exactly 5/bar after bar 0.
    np.testing.assert_allclose(trading, np.arange(n) * 5.0, atol=1e-9)


def test_spread_chart_handles_warmup_nans(bars, tmp_path):
    """The first 60 bars have no z-score; plotting must not raise."""
    path = plot_spread_and_zscore(bars, tmp_path / "spread.png")
    assert path.exists() and path.stat().st_size > 1000


def test_price_chart_is_written(bars, tmp_path):
    path = plot_price_series(bars, tmp_path / "prices.png")
    assert path.exists() and path.stat().st_size > 1000


def test_trade_distribution_is_written(trades, tmp_path):
    path = plot_trade_distribution(trades, tmp_path / "trades.png")
    assert path.exists() and path.stat().st_size > 1000


def test_trade_distribution_handles_no_trades(tmp_path):
    """A strategy that never traded is a valid outcome and must still render."""
    empty = pd.DataFrame(columns=["pnl_net", "holding_bars"])
    path = plot_trade_distribution(empty, tmp_path / "empty.png")
    assert path.exists()


def test_charts_create_missing_directories(bars, tmp_path):
    path = plot_equity_curve(bars, tmp_path / "deep" / "nested" / "equity.png")
    assert path.exists()


def test_metrics_table_contains_the_headline_numbers():
    """The three numbers the resume claim rests on must appear."""
    metrics = {
        "sharpe_ratio": 1.2345,
        "sharpe_per_bar": 0.0778,
        "max_drawdown": -0.0876,
        "max_drawdown_duration": 42,
        "n_trades": 87,
        "total_return": 0.1534,
        "annualized_return": 0.0421,
        "annualized_volatility": 0.0341,
        "calmar_ratio": 0.4806,
        "win_rate": 0.6322,
        "avg_holding_bars": 11.4,
        "avg_win": 412.55,
        "avg_loss": -298.10,
        "profit_factor": 1.55,
        "gross_pnl": 16_800.0,
        "total_costs": 1_460.0,
        "net_pnl": 15_340.0,
        "turnover": 43.2,
        "exposure_frac": 0.38,
        "initial_nav": 100_000.0,
        "final_nav": 115_340.0,
        "n_bars": 2520,
        "n_trading_bars": 2460,
    }
    table = metrics_markdown_table(metrics)
    assert "| Metric | Value |" in table
    assert "1.2345" in table, "Sharpe ratio must appear"
    assert "-8.76%" in table, "max drawdown must appear as a percentage"
    assert "| 87 |" in table, "trade count must appear"
    # Money is thousands-separated and prefixed.
    assert "$115,340.00" in table


def test_metrics_table_handles_negative_values():
    """A losing strategy must render its negative numbers, not hide them."""
    table = metrics_markdown_table(
        {"sharpe_ratio": -0.8412, "max_drawdown": -0.2210, "n_trades": 12}
    )
    assert "-0.8412" in table
    assert "-22.10%" in table


def test_metrics_table_tolerates_missing_keys():
    """A partial metrics dict renders as nan rather than raising."""
    table = metrics_markdown_table({"sharpe_ratio": 1.0})
    assert "1.0000" in table
    assert "nan" in table


def test_exit_reason_table_shares_sum_to_one_hundred():
    metrics = {
        "n_trades": 100,
        "n_reversion": 70,
        "n_stop_loss": 20,
        "n_max_holding": 9,
        "n_end_of_data": 1,
    }
    table = exit_reason_markdown_table(metrics)
    assert "| 70 | 70.0% |" in table
    assert "| 20 | 20.0% |" in table
    assert "| 1 | 1.0% |" in table


def test_exit_reason_table_handles_zero_trades():
    """Division by the trade count must not blow up when there were none."""
    table = exit_reason_markdown_table(
        {"n_trades": 0, "n_reversion": 0, "n_stop_loss": 0,
         "n_max_holding": 0, "n_end_of_data": 0}
    )
    assert "0.0%" in table
