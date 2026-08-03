"""End-to-end tests across the OCaml/Python boundary.

These are the tests that would catch a schema drift between the two languages —
the failure mode where both sides pass their own unit tests but disagree about
the file they exchange.

They are skipped, not failed, when the engine has not been built, so that
``pytest`` alone is useful without a working OCaml toolchain. ``make test``
builds first, so CI always exercises them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statarb.io import read_bars, read_metrics, read_prices, read_trades, write_prices
from statarb.synth import CointegratedSpec, generate_cointegrated

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENGINE_BINARY = REPO_ROOT / "engine" / "_build" / "default" / "bin" / "main.exe"

requires_engine = pytest.mark.skipif(
    not ENGINE_BINARY.exists(),
    reason=f"engine not built at {ENGINE_BINARY}; run 'make build'",
)


def run_backtest(prices_csv: Path, out_dir: Path, *extra: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(ENGINE_BINARY), "backtest",
            "--prices", str(prices_csv),
            "--out-dir", str(out_dir),
            "--quiet", *extra,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"engine failed:\n{result.stderr}"


@pytest.fixture
def cointegrated_csv(tmp_path) -> Path:
    spec = CointegratedSpec(
        n_bars=800, beta=1.2, half_life=15.0, sigma_spread=0.03, sigma_common=0.012
    )
    frame = generate_cointegrated(spec, seed=101)
    return write_prices(frame, tmp_path / "prices.csv")


@requires_engine
def test_python_writes_what_ocaml_reads(cointegrated_csv, tmp_path):
    """The schema contract holds: a file Python writes, OCaml consumes."""
    out_dir = tmp_path / "out"
    run_backtest(cointegrated_csv, out_dir)
    assert (out_dir / "bars.csv").exists()
    assert (out_dir / "trades.csv").exists()
    assert (out_dir / "metrics.csv").exists()


@requires_engine
def test_bar_count_matches_input(cointegrated_csv, tmp_path):
    """One output bar per input bar — no silent truncation of the sample."""
    out_dir = tmp_path / "out"
    run_backtest(cointegrated_csv, out_dir)
    assert len(read_bars(out_dir / "bars.csv")) == len(read_prices(cointegrated_csv))


@requires_engine
def test_nav_identity_holds_in_the_output(cointegrated_csv, tmp_path):
    """NAV = cash + position value, verified independently in Python.

    The engine asserts this internally, but checking from the outside against
    the emitted file is what a third party would do — and it is what catches a
    bug in the *writer* as opposed to in the accounting.
    """
    out_dir = tmp_path / "out"
    run_backtest(cointegrated_csv, out_dir)
    bars = read_bars(out_dir / "bars.csv")
    np.testing.assert_allclose(
        bars["cash"] + bars["position_value"], bars["nav"], atol=1e-6
    )


@requires_engine
def test_nav_evolves_only_by_mark_to_market_interest_and_costs(
    cointegrated_csv, tmp_path
):
    """The accounting identity, re-derived in Python from the audit trail.

    NAV moves only through mark-to-market on the position held into the bar,
    interest accrued on cash, and costs paid. Nothing else.
    """
    out_dir = tmp_path / "out"
    run_backtest(cointegrated_csv, out_dir)
    bars = read_bars(out_dir / "bars.csv")

    prev = bars.iloc[:-1].reset_index(drop=True)
    cur = bars.iloc[1:].reset_index(drop=True)
    mtm = prev["qty_a"] * (cur["price_a"] - prev["price_a"]) + prev["qty_b"] * (
        cur["price_b"] - prev["price_b"]
    )
    expected = prev["nav"] + mtm + cur["interest_this_bar"] - cur["costs_this_bar"]
    np.testing.assert_allclose(expected, cur["nav"], atol=1e-6)


@requires_engine
def test_entries_fill_one_bar_after_the_signal(cointegrated_csv, tmp_path):
    """Next-bar execution, verified from the output file.

    For every entry fill, the *previous* bar's z-score must have breached the
    entry threshold. If fills happened on the signal bar, this would fail.
    """
    out_dir = tmp_path / "out"
    run_backtest(cointegrated_csv, out_dir, "--entry", "2.0")
    bars = read_bars(out_dir / "bars.csv")

    entry_rows = bars.index[bars["trade_event"].str.startswith("entry")].tolist()
    assert entry_rows, "expected at least one entry to verify"
    for i in entry_rows:
        assert i > 0, "an entry on bar 0 would be impossible"
        previous_z = bars.loc[i - 1, "zscore"]
        assert not np.isnan(previous_z), f"entry at {i} but no z-score at {i - 1}"
        assert abs(previous_z) >= 2.0 - 1e-9, (
            f"entry at bar {i} but the previous bar's z-score was {previous_z:.4f}, "
            "which did not breach the threshold"
        )


@requires_engine
def test_trade_pnl_reconciles_with_metrics(cointegrated_csv, tmp_path):
    """The trade log and the metrics file must agree.

    Two independent summaries of the same run; a disagreement means one of them
    is computed from stale or separate state.
    """
    out_dir = tmp_path / "out"
    run_backtest(cointegrated_csv, out_dir)
    trades = read_trades(out_dir / "trades.csv")
    metrics = read_metrics(out_dir / "metrics.csv")

    assert len(trades) == int(metrics["n_trades"])
    if len(trades) > 0:
        np.testing.assert_allclose(
            trades["pnl_net"], trades["pnl_gross"] - trades["costs"], atol=1e-9
        )
        assert int((trades["pnl_net"] > 0).sum()) == int(metrics["n_wins"])
        # Net PnL over the whole run is the NAV change.
        assert metrics["net_pnl"] == pytest.approx(
            metrics["final_nav"] - metrics["initial_nav"], abs=1e-6
        )


@requires_engine
def test_metrics_are_internally_consistent(cointegrated_csv, tmp_path):
    """Cross-check the metrics against each other and against the NAV series."""
    out_dir = tmp_path / "out"
    run_backtest(cointegrated_csv, out_dir)
    metrics = read_metrics(out_dir / "metrics.csv")
    bars = read_bars(out_dir / "bars.csv")

    assert metrics["initial_nav"] == pytest.approx(bars["nav"].iloc[0], abs=1e-6)
    assert metrics["final_nav"] == pytest.approx(bars["nav"].iloc[-1], abs=1e-6)
    assert metrics["total_return"] == pytest.approx(
        metrics["final_nav"] / metrics["initial_nav"] - 1.0, abs=1e-9
    )
    assert metrics["max_drawdown"] <= 0.0, "max drawdown must be reported as negative"

    # Recompute max drawdown independently from the NAV series.
    nav = bars["nav"].to_numpy()
    peak = np.maximum.accumulate(nav)
    assert metrics["max_drawdown"] == pytest.approx(float(np.min(nav / peak - 1.0)), abs=1e-8)

    assert metrics["n_wins"] + metrics["n_losses"] == metrics["n_trades"]
    exits = (
        metrics["n_reversion"] + metrics["n_stop_loss"]
        + metrics["n_max_holding"] + metrics["n_end_of_data"]
    )
    assert exits == metrics["n_trades"], "every trade must have exactly one exit reason"


@requires_engine
def test_engine_is_deterministic(cointegrated_csv, tmp_path):
    """The same input produces byte-identical output.

    Reproducibility is a claim the README makes; this is the test of it.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    run_backtest(cointegrated_csv, a)
    run_backtest(cointegrated_csv, b)
    for name in ("bars.csv", "trades.csv", "metrics.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), f"{name} differs"


@requires_engine
def test_higher_costs_reduce_final_nav(cointegrated_csv, tmp_path):
    """Cost monotonicity, across the CLI boundary."""
    cheap, dear = tmp_path / "cheap", tmp_path / "dear"
    run_backtest(cointegrated_csv, cheap, "--commission-bps", "0", "--slippage-bps", "0")
    run_backtest(cointegrated_csv, dear, "--commission-bps", "10", "--slippage-bps", "10")
    assert read_metrics(cheap / "metrics.csv")["total_costs"] == pytest.approx(0.0)
    assert (
        read_metrics(dear / "metrics.csv")["final_nav"]
        <= read_metrics(cheap / "metrics.csv")["final_nav"] + 1e-9
    )


@requires_engine
def test_engine_rejects_a_malformed_file(tmp_path):
    """A corrupt input is a loud failure, not a silently shortened sample."""
    bad = tmp_path / "bad.csv"
    bad.write_text("date,price_a,price_b\n2020-01-01,100.0\n")
    result = subprocess.run(
        [str(ENGINE_BINARY), "backtest", "--prices", str(bad),
         "--out-dir", str(tmp_path / "out"), "--quiet"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "parse error" in (result.stdout + result.stderr).lower()


@requires_engine
def test_engine_rejects_an_incoherent_config(cointegrated_csv, tmp_path):
    """exit >= entry would open and close a position every bar."""
    result = subprocess.run(
        [str(ENGINE_BINARY), "backtest", "--prices", str(cointegrated_csv),
         "--out-dir", str(tmp_path / "out"), "--entry", "1.0", "--exit", "2.0"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "exit_threshold" in (result.stdout + result.stderr)


@requires_engine
def test_engine_rejects_an_unknown_flag(cointegrated_csv, tmp_path):
    """A typo in the Makefile must not silently run different parameters."""
    result = subprocess.run(
        [str(ENGINE_BINARY), "backtest", "--prices", str(cointegrated_csv),
         "--out-dir", str(tmp_path / "out"), "--entrry", "2.0"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "unknown flag" in (result.stdout + result.stderr).lower()


@requires_engine
def test_negative_control_produces_no_real_edge(tmp_path):
    """Two independent random walks must not yield a strong Sharpe.

    This is the honesty check at the whole-system level. If the backtester
    reported an impressive Sharpe on data with no exploitable relationship, the
    machinery would be manufacturing profit — and every other number it
    produces would be suspect.

    The bound is loose (|Sharpe| < 1.5) because a single sample path of a
    no-edge strategy has substantial sampling variation; the assertion is that
    the result is not *dramatic*, which is what a lookahead bug would produce.
    """
    from statarb.synth import generate_independent

    frame = generate_independent(2000, seed=202)
    prices = write_prices(frame, tmp_path / "noise.csv")
    out_dir = tmp_path / "out"
    run_backtest(prices, out_dir)
    sharpe = read_metrics(out_dir / "metrics.csv")["sharpe_ratio"]
    assert abs(sharpe) < 1.5, (
        f"a no-edge dataset produced Sharpe {sharpe:.4f}; "
        "this suggests the backtester is manufacturing an edge"
    )
