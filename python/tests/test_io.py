"""Tests for the CSV interchange layer.

These tests are mostly about *refusal*. The engine's parser is strict, so the
writer must be too — a frame that the writer accepts but the engine rejects
would fail halfway through the pipeline with a message pointing at the wrong
stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb import PRICE_COLUMNS
from statarb.io import read_bars, read_metrics, read_prices, read_trades, write_prices


def _good_frame(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d"),
            "price_a": np.linspace(100.0, 110.0, n),
            "price_b": np.linspace(50.0, 52.0, n),
        }
    )


def test_write_and_read_round_trip(tmp_path):
    path = write_prices(_good_frame(), tmp_path / "prices.csv")
    back = read_prices(path)
    assert list(back.columns) == list(PRICE_COLUMNS)
    assert len(back) == 10
    assert back["price_a"].iloc[0] == pytest.approx(100.0)


def test_write_drops_extra_columns(tmp_path):
    """Ground-truth columns must never reach the engine's input file.

    The engine seeing the true spread would be the ultimate lookahead, and its
    parser would reject the extra field anyway — so dropping them here is both
    a correctness and a compatibility requirement.
    """
    frame = _good_frame()
    frame["true_spread"] = np.random.default_rng(0).normal(size=len(frame))
    frame["common_trend"] = 1.0
    path = write_prices(frame, tmp_path / "prices.csv")
    header = path.read_text().splitlines()[0]
    assert header == "date,price_a,price_b"


def test_write_rejects_missing_columns(tmp_path):
    frame = _good_frame().drop(columns=["price_b"])
    with pytest.raises(ValueError, match="missing columns"):
        write_prices(frame, tmp_path / "prices.csv")


def test_write_rejects_non_positive_prices(tmp_path):
    frame = _good_frame()
    frame.loc[3, "price_a"] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        write_prices(frame, tmp_path / "prices.csv")


def test_write_rejects_non_finite_prices(tmp_path):
    frame = _good_frame()
    frame.loc[3, "price_b"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        write_prices(frame, tmp_path / "prices.csv")


def test_write_rejects_out_of_order_dates(tmp_path):
    frame = _good_frame()
    frame.loc[3, "date"] = "1999-01-01"
    with pytest.raises(ValueError, match="strictly increasing"):
        write_prices(frame, tmp_path / "prices.csv")


def test_write_rejects_duplicate_dates(tmp_path):
    frame = _good_frame()
    frame.loc[3, "date"] = frame.loc[2, "date"]
    with pytest.raises(ValueError, match="strictly increasing"):
        write_prices(frame, tmp_path / "prices.csv")


def test_write_rejects_an_empty_frame(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_prices(_good_frame(0), tmp_path / "prices.csv")


def test_write_is_byte_reproducible(tmp_path):
    """Two writes of the same frame must be byte-identical.

    This is what lets `make backtest` regenerate the README's inputs exactly
    rather than merely equivalently.
    """
    frame = _good_frame(50)
    a = write_prices(frame, tmp_path / "a.csv").read_bytes()
    b = write_prices(frame, tmp_path / "b.csv").read_bytes()
    assert a == b


def test_read_bars_normalises_optional_fields(tmp_path):
    """Warm-up bars have empty statistics; they must read as NaN, not 0."""
    path = tmp_path / "bars.csv"
    path.write_text(
        "index,date,price_a,price_b,hedge_ratio,spread,zscore,position,qty_a,"
        "qty_b,cash,position_value,nav,costs_this_bar,interest_this_bar,trade_event\n"
        "0,2020-01-01,100.0,50.0,,,,flat,0.0,0.0,100000.0,0.0,100000.0,0.0,0.0,\n"
        "1,2020-01-02,101.0,50.5,1.25,0.01,-2.1,long_spread,250.0,-500.0,"
        "99992.5,0.0,99992.5,7.5,15.87,entry:long_spread\n"
    )
    bars = read_bars(path)
    assert np.isnan(bars["zscore"].iloc[0]), "an empty statistic must be NaN, not 0"
    assert bars["zscore"].iloc[1] == pytest.approx(-2.1)
    # An absent trade_event normalises to "" so string ops need no null check.
    assert bars["trade_event"].iloc[0] == ""
    assert bars["trade_event"].iloc[1] == "entry:long_spread"


def test_read_bars_rejects_a_missing_column(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text("index,date,nav\n0,2020-01-01,100000.0\n")
    with pytest.raises(ValueError, match="missing bar columns"):
        read_bars(path)


def test_read_metrics_parses_numbers(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("metric,value\nsharpe_ratio,1.2345\nn_trades,42\n")
    metrics = read_metrics(path)
    assert metrics["sharpe_ratio"] == pytest.approx(1.2345)
    assert metrics["n_trades"] == pytest.approx(42.0)


def test_read_metrics_rejects_a_bad_schema(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("name,val\nsharpe,1.0\n")
    with pytest.raises(ValueError, match="metric.*value"):
        read_metrics(path)


def test_read_trades_handles_an_empty_log(tmp_path):
    """A strategy that never traded is a valid outcome, not an error."""
    path = tmp_path / "trades.csv"
    path.write_text(
        "direction,entry_index,exit_index,entry_date,exit_date,entry_z,exit_z,"
        "pnl_gross,costs,pnl_net,exit_reason,holding_bars\n"
    )
    assert len(read_trades(path)) == 0
