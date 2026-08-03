#!/usr/bin/env python3
"""Full reproducible pipeline: generate data, backtest, analyse, report.

Run via ``make backtest``. Every number in the README is produced by this
script; nothing is hand-written. Re-running it from a clean checkout must
reproduce the reported figures exactly, which is what makes the claims
checkable by a third party.

Stages
------
1. Generate the synthetic datasets from their fixed seeds.
2. Run the cointegration battery on each and write the diagnostics.
3. Run the OCaml engine on each.
4. Run the parameter sweep on the primary dataset.
5. Attempt real market data; record success or the reason for failure.
6. Emit charts and the machine-readable summary the README is built from.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import pandas as pd  # noqa: E402

from statarb import io as sio  # noqa: E402
from statarb import report  # noqa: E402
from statarb.cointegration import analyse_pair  # noqa: E402
from statarb.datasets import SYNTHETIC_DATASETS, Dataset  # noqa: E402
from statarb.fetch import CANDIDATE_PAIRS, fetch_pair  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = REPO_ROOT / "reports"
ENGINE_DIR = REPO_ROOT / "engine"

#: The engine binary, relative to the engine directory, as dune builds it.
ENGINE_BINARY = ENGINE_DIR / "_build" / "default" / "bin" / "main.exe"


@dataclass
class RunOutcome:
    """Result of backtesting one dataset."""

    key: str
    title: str
    is_negative_control: bool
    metrics: dict = field(default_factory=dict)
    cointegration: dict = field(default_factory=dict)
    charts: dict = field(default_factory=dict)
    error: str = ""


def log(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


def run_engine(prices_csv: Path, out_dir: Path, extra_args: list[str] | None = None) -> None:
    """Invoke the OCaml engine.

    Raises:
        RuntimeError: If the binary is missing or exits non-zero. A failed
            backtest must abort the pipeline rather than leaving stale outputs
            in place for the report stage to pick up as if they were fresh.
    """
    if not ENGINE_BINARY.exists():
        raise RuntimeError(
            f"engine binary not found at {ENGINE_BINARY}; run 'make build' first"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ENGINE_BINARY),
        "backtest",
        "--prices",
        str(prices_csv),
        "--out-dir",
        str(out_dir),
        "--quiet",
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"engine failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def run_sweep(prices_csv: Path, out_csv: Path, label: str) -> None:
    """Run the parameter sensitivity sweep."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(ENGINE_BINARY),
            "sweep",
            "--prices",
            str(prices_csv),
            "--out",
            str(out_csv),
            "--label",
            label,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sweep failed:\n{result.stderr}")


def process_dataset(dataset: Dataset) -> RunOutcome:
    """Generate, analyse, backtest, and chart one dataset."""
    log(f"dataset '{dataset.key}': generating {dataset.n_bars} bars")
    frame = dataset.generate()

    prices_csv = RAW_DIR / f"{dataset.key}.csv"
    sio.write_prices(frame, prices_csv)

    outcome = RunOutcome(
        key=dataset.key,
        title=dataset.title,
        is_negative_control=dataset.is_negative_control,
    )

    # Cointegration diagnostics. Run on the same prices the engine sees.
    log(f"dataset '{dataset.key}': cointegration battery")
    stats = analyse_pair(frame["price_a"], frame["price_b"], name_a="A", name_b="B")
    outcome.cointegration = stats.to_dict()
    if dataset.spec is not None:
        # Ground truth is recorded so the README can show the estimator
        # recovering the parameters we chose.
        outcome.cointegration["true_beta"] = dataset.spec.beta
        outcome.cointegration["true_half_life"] = dataset.spec.half_life
        outcome.cointegration["true_sigma_spread"] = dataset.spec.sigma_spread

    # Backtest.
    log(f"dataset '{dataset.key}': running engine")
    out_dir = REPORTS_DIR / dataset.key
    try:
        run_engine(prices_csv, out_dir)
    except RuntimeError as exc:
        outcome.error = str(exc)
        return outcome

    outcome.metrics = sio.read_metrics(out_dir / "metrics.csv")

    # Charts.
    log(f"dataset '{dataset.key}': charts")
    bars = sio.read_bars(out_dir / "bars.csv")
    trades = sio.read_trades(out_dir / "trades.csv")
    outcome.charts = {
        "equity": str(
            report.plot_equity_curve(bars, out_dir / "equity_curve.png").relative_to(
                REPO_ROOT
            )
        ),
        "spread": str(
            report.plot_spread_and_zscore(
                bars, out_dir / "spread_zscore.png"
            ).relative_to(REPO_ROOT)
        ),
        "prices": str(
            report.plot_price_series(bars, out_dir / "prices_hedge.png").relative_to(
                REPO_ROOT
            )
        ),
        "trades": str(
            report.plot_trade_distribution(
                trades, out_dir / "trade_distribution.png"
            ).relative_to(REPO_ROOT)
        ),
    }
    return outcome


def process_real_data() -> dict:
    """Attempt the real-data section. Never fatal.

    Returns a dict recording either the results or the reason none are
    available, so the README can state plainly which happened.
    """
    log("attempting real market data via yfinance")
    attempts: list[dict] = []
    successes: list[dict] = []

    for ticker_a, ticker_b, sector in CANDIDATE_PAIRS:
        result = fetch_pair(ticker_a, ticker_b)
        if not result.ok:
            attempts.append(
                {"pair": f"{ticker_a}/{ticker_b}", "sector": sector, "error": result.reason}
            )
            continue

        frame = result.frame
        assert frame is not None
        key = f"real_{ticker_a}_{ticker_b}"
        prices_csv = RAW_DIR / f"{key}.csv"
        try:
            sio.write_prices(frame, prices_csv)
            stats = analyse_pair(
                frame["price_a"], frame["price_b"], name_a=ticker_a, name_b=ticker_b
            )
            out_dir = REPORTS_DIR / key
            run_engine(prices_csv, out_dir)
            metrics = sio.read_metrics(out_dir / "metrics.csv")
            bars = sio.read_bars(out_dir / "bars.csv")
            trades = sio.read_trades(out_dir / "trades.csv")
            charts = {
                "equity": str(
                    report.plot_equity_curve(
                        bars, out_dir / "equity_curve.png"
                    ).relative_to(REPO_ROOT)
                ),
                "spread": str(
                    report.plot_spread_and_zscore(
                        bars, out_dir / "spread_zscore.png"
                    ).relative_to(REPO_ROOT)
                ),
                "trades": str(
                    report.plot_trade_distribution(
                        trades, out_dir / "trade_distribution.png"
                    ).relative_to(REPO_ROOT)
                ),
            }
            successes.append(
                {
                    "key": key,
                    "pair": f"{ticker_a}/{ticker_b}",
                    "sector": sector,
                    "n_bars": len(frame),
                    "start": frame["date"].iloc[0],
                    "end": frame["date"].iloc[-1],
                    "cointegration": stats.to_dict(),
                    "metrics": metrics,
                    "charts": charts,
                }
            )
            attempts.append({"pair": f"{ticker_a}/{ticker_b}", "sector": sector, "error": ""})
            log(f"real data: {ticker_a}/{ticker_b} succeeded ({len(frame)} bars)")
        except Exception as exc:  # noqa: BLE001 - record and continue to the next pair
            attempts.append(
                {"pair": f"{ticker_a}/{ticker_b}", "sector": sector, "error": str(exc)}
            )
            log(f"real data: {ticker_a}/{ticker_b} failed: {exc}")

    return {
        "available": len(successes) > 0,
        "attempts": attempts,
        "results": successes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-real-data",
        action="store_true",
        help="skip the network fetch (useful offline or in CI)",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    outcomes: list[RunOutcome] = []
    for dataset in SYNTHETIC_DATASETS:
        outcome = process_dataset(dataset)
        if outcome.error:
            log(f"ERROR on '{dataset.key}': {outcome.error}")
            return 1
        outcomes.append(outcome)

    # Sensitivity sweep on the primary dataset.
    primary = SYNTHETIC_DATASETS[0]
    log(f"parameter sweep on '{primary.key}'")
    sweep_csv = REPORTS_DIR / "sweep.csv"
    run_sweep(RAW_DIR / f"{primary.key}.csv", sweep_csv, primary.key)

    real = (
        {"available": False, "attempts": [], "results": [], "skipped": True}
        if args.skip_real_data
        else process_real_data()
    )

    summary = {
        "synthetic": [asdict(o) for o in outcomes],
        "sweep_csv": str(sweep_csv.relative_to(REPO_ROOT)),
        "real_data": real,
    }
    summary_path = REPORTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log(f"wrote {summary_path.relative_to(REPO_ROOT)}")

    # Human-readable console summary. The authoritative numbers live in the
    # CSVs; this is a convenience view of the same values.
    print()
    print("=" * 78)
    print("PIPELINE RESULTS")
    print("=" * 78)
    for outcome in outcomes:
        marker = "  [NEGATIVE CONTROL]" if outcome.is_negative_control else ""
        print(f"\n{outcome.title}{marker}")
        m = outcome.metrics
        print(
            f"  Sharpe {m.get('sharpe_ratio', float('nan')):+.4f}   "
            f"max DD {m.get('max_drawdown', float('nan')) * 100:+.2f}%   "
            f"trades {int(m.get('n_trades', 0))}   "
            f"total return {m.get('total_return', float('nan')) * 100:+.2f}%"
        )
        c = outcome.cointegration
        if "true_beta" in c:
            print(
                f"  beta: true {c['true_beta']:.3f}, "
                f"full-sample estimate {c['hedge_ratio']:.3f}   "
                f"half-life: true {c['true_half_life']:.1f}, "
                f"estimated {c['half_life']:.1f} bars"
            )
        print(
            f"  cointegration: EG p={c['eg_pvalue']:.5f}  "
            f"ADF p={c['adf_pvalue']:.5f}  "
            f"Johansen {c['johansen_trace_stat']:.2f} vs crit {c['johansen_crit_95']:.2f}  "
            f"-> {'COINTEGRATED' if c['is_cointegrated'] else 'NOT cointegrated'}"
        )
    print()
    if real.get("skipped"):
        print("Real data: skipped (--skip-real-data)")
    elif real["available"]:
        print(f"Real data: {len(real['results'])} pair(s) succeeded")
        for r in real["results"]:
            print(
                f"  {r['pair']:>10}  Sharpe {r['metrics'].get('sharpe_ratio', float('nan')):+.4f}  "
                f"max DD {r['metrics'].get('max_drawdown', float('nan')) * 100:+.2f}%  "
                f"trades {int(r['metrics'].get('n_trades', 0))}"
            )
    else:
        print("Real data: UNAVAILABLE")
        for a in real["attempts"][:3]:
            print(f"  {a['pair']}: {a['error']}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
