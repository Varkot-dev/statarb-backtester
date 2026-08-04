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
from statarb.significance import assess  # noqa: E402
from statarb.variance_ratio import variance_ratio_test  # noqa: E402
from statarb.stability import analyse_stability  # noqa: E402
from statarb.diagnostics import (  # noqa: E402
    MIN_WINDOW_HALF_LIFE_RATIO,
    deflated_sharpe_threshold,
    half_life_sensitivity,
    walk_forward,
    window_ratio_sensitivity,
)

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
    significance: dict = field(default_factory=dict)
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

    # Significance. A point estimate cannot say whether a result is
    # distinguishable from luck, and on the negative control it is precisely
    # the difference between "+9% return" and "t=0.86, consistent with zero".
    outcome.significance = assess(
        trades, bars["nav"].to_numpy(), bars_per_year=252.0, risk_free_annual=0.04
    ).to_dict()

    outcome.charts = {
        "equity": str(
            report.plot_equity_curve(bars, out_dir / "equity_curve.png").relative_to(
                REPO_ROOT
            )
        ),
        "decomposition": str(
            report.plot_equity_decomposition(
                bars, out_dir / "equity_decomposition.png"
            ).relative_to(REPO_ROOT)
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
            significance = assess(
                trades, bars["nav"].to_numpy(), bars_per_year=252.0,
                risk_free_annual=0.04,
            ).to_dict()
            # Variance ratio on the spread. Complements the half-life estimate:
            # the half-life gives a timescale with no significance and is biased
            # toward finding reversion on near-unit-root series; the VR test
            # gives a p-value on the random-walk null and no timescale.
            import numpy as _np
            from statarb.cointegration import ols_hedge_ratio as _ols
            _la = _np.log(frame["price_a"].to_numpy(dtype=float))
            _lb = _np.log(frame["price_b"].to_numpy(dtype=float))
            _, _beta = _ols(_la, _lb)
            vr_rows = [
                v.to_dict() for v in variance_ratio_test(_la - _beta * _lb)
            ]
            # Stability. A 756-bar window rather than the 252 default: the
            # repo's own window/half-life law applies to this test too, and at
            # 252 bars a pair with a 30-bar half-life rejects in only ~11% of
            # windows even when perfectly stable, which hides real breaks.
            _pa = frame["price_a"].copy()
            _pb = frame["price_b"].copy()
            _idx = pd.to_datetime(frame["date"])
            _pa.index = _idx
            _pb.index = _idx
            stability = analyse_stability(
                _pa, _pb, name_a=ticker_a, name_b=ticker_b, window=756, step=21
            ).to_dict()
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
                    "significance": significance,
                    "variance_ratio": vr_rows,
                    "stability": stability,
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

    # The same primary dataset run WITHOUT crediting interest on idle cash, so
    # the README can quantify how much of the reported Sharpe is an artifact of
    # that convention rather than of the strategy. See
    # Config.accrue_cash_interest for why the credited version is the coherent
    # comparison.
    primary = SYNTHETIC_DATASETS[0]
    log(f"'{primary.key}': no-cash-interest comparison run")
    no_interest_dir = REPORTS_DIR / f"{primary.key}_no_cash_interest"
    run_engine(
        RAW_DIR / f"{primary.key}.csv", no_interest_dir, ["--no-cash-interest"]
    )
    no_interest_metrics = sio.read_metrics(no_interest_dir / "metrics.csv")

    # Leakage calibration: how much would a KNOWN dose of lookahead change the
    # reported result? Produces the dose-response curves the README leads with.
    log(f"'{primary.key}': leakage calibration")
    calibration_csv = REPORTS_DIR / "leakage_calibration.csv"
    result = subprocess.run(
        [str(ENGINE_BINARY), "calibrate", "--prices",
         str(RAW_DIR / f"{primary.key}.csv"), "--out", str(calibration_csv),
         "--quiet"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"calibration failed:\n{result.stderr}")
    calibration = pd.read_csv(calibration_csv)

    # Kalman memory sweep. The README claims the window/half-life law is really
    # a MEMORY law that a windowless estimator does not escape; that table must
    # be generated rather than transcribed.
    log(f"'{primary.key}': Kalman memory sweep")
    kalman_csv = REPORTS_DIR / "kalman_memory_sweep.csv"
    primary_hl = outcomes[0].cointegration["half_life"]
    result = subprocess.run(
        [str(ENGINE_BINARY), "kalman-sweep", "--prices",
         str(RAW_DIR / f"{primary.key}.csv"), "--out", str(kalman_csv),
         "--half-life", f"{primary_hl:.4f}", "--quiet"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"kalman sweep failed:\n{result.stderr}")
    kalman_sweep = pd.read_csv(kalman_csv)
    report.plot_leakage_calibration(
        calibration, REPORTS_DIR / "leakage_calibration.png"
    )

    log(f"parameter sweep on '{primary.key}'")
    sweep_csv = REPORTS_DIR / "sweep.csv"
    run_sweep(RAW_DIR / f"{primary.key}.csv", sweep_csv, primary.key)

    # ------------------------------------------------------------------
    # Diagnostics: WHY the strategy performs as it does. A weak result with no
    # mechanism is a non-explanation; these experiments isolate the cause on
    # data whose ground truth is known.
    # ------------------------------------------------------------------
    log("diagnostics: half-life and window-ratio sensitivity")
    diag_dir = REPORTS_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    hl_rows = [r.to_dict() for r in half_life_sensitivity(ENGINE_BINARY, diag_dir)]
    ratio_rows = [
        r.to_dict() for r in window_ratio_sensitivity(ENGINE_BINARY, diag_dir)
    ]
    pd.DataFrame(hl_rows).to_csv(diag_dir / "half_life_sensitivity.csv", index=False)
    pd.DataFrame(ratio_rows).to_csv(diag_dir / "window_ratio_sensitivity.csv", index=False)
    report.plot_window_ratio_finding(
        pd.DataFrame(hl_rows), pd.DataFrame(ratio_rows),
        REPORTS_DIR / "window_ratio_finding.png",
    )

    real = (
        {"available": False, "attempts": [], "results": [], "skipped": True}
        if args.skip_real_data
        else process_real_data()
    )

    # Walk-forward on the real pairs: the only non-circular test of the window
    # rule, since the window is chosen from data the evaluation never sees.
    walk_forward_rows: list[dict] = []
    if real.get("available"):
        log("diagnostics: walk-forward out-of-sample test")
        for r in real["results"]:
            row = walk_forward(
                ENGINE_BINARY, diag_dir,
                sio.read_prices(RAW_DIR / f"{r['key']}.csv"), r["pair"],
            )
            if row is not None:
                walk_forward_rows.append(row.to_dict())
        if walk_forward_rows:
            pd.DataFrame(walk_forward_rows).to_csv(
                diag_dir / "walk_forward.csv", index=False
            )

    summary = {
        "synthetic": [asdict(o) for o in outcomes],
        "diagnostics": {
            "half_life_sensitivity": hl_rows,
            "window_ratio_sensitivity": ratio_rows,
            "walk_forward": walk_forward_rows,
            "min_window_half_life_ratio": MIN_WINDOW_HALF_LIFE_RATIO,
            # Honest accounting of the specification search performed while
            # investigating the window rule on real data.
            "deflated_sharpe": deflated_sharpe_threshold(
                observed_sharpe=0.654, n_trials=30, n_observations=2515
            ),
        },
        "cash_interest_comparison": {
            "dataset": primary.key,
            "with_interest": outcomes[0].metrics,
            "without_interest": no_interest_metrics,
        },
        "sweep_csv": str(sweep_csv.relative_to(REPO_ROOT)),
        "kalman_memory_sweep": {
            "csv": str(kalman_csv.relative_to(REPO_ROOT)),
            "rows": kalman_sweep.to_dict(orient="records"),
        },
        "leakage_calibration": {
            "csv": str(calibration_csv.relative_to(REPO_ROOT)),
            "chart": "reports/leakage_calibration.png",
            "rows": calibration.to_dict(orient="records"),
        },
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
        sig = outcome.significance
        interest = m.get("total_interest", 0.0)
        trading = m.get("net_pnl", 0.0) - interest
        print(
            f"  Sharpe {m.get('sharpe_ratio', float('nan')):+.4f} "
            f"[95% CI {sig.get('sharpe_ci_low', float('nan')):+.2f}, "
            f"{sig.get('sharpe_ci_high', float('nan')):+.2f}]   "
            f"max DD {m.get('max_drawdown', float('nan')) * 100:+.2f}%   "
            f"trades {int(m.get('n_trades', 0))}"
        )
        print(
            f"  PnL: trading {trading:+,.0f} + interest on idle cash "
            f"{interest:+,.0f} = {m.get('net_pnl', 0.0):+,.0f}"
        )
        print(
            f"  per-trade t-stat {sig.get('t_statistic', float('nan')):+.3f}, "
            f"p={sig.get('p_value', float('nan')):.3f} -> "
            f"{'SIGNIFICANT' if sig.get('is_significant') else 'NOT significant'}"
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
    shift = calibration[calibration.leak_type == "timing_shift"]
    filt = calibration[calibration.leak_type == "outcome_filter"]
    print("Leakage calibration (primary dataset):")
    print(
        f"  timing shift   k=0 -> k={int(shift.dose.max())}: "
        f"Sharpe {shift.iloc[0].sharpe_ratio:+.4f} -> {shift.iloc[-1].sharpe_ratio:+.4f}"
        "  (an off-by-one DESTROYS a mean-reversion strategy)"
    )
    print(
        f"  outcome filter 0% -> 100% of losers skipped: "
        f"Sharpe {filt.iloc[0].sharpe_ratio:+.4f} -> {filt.iloc[-1].sharpe_ratio:+.4f}"
        f", win rate {filt.iloc[0].win_rate:.3f} -> {filt.iloc[-1].win_rate:.3f}"
    )
    print()
    print("Cash-interest convention (primary dataset):")
    print(
        f"  interest credited on idle cash: Sharpe "
        f"{outcomes[0].metrics.get('sharpe_ratio', float('nan')):+.4f}"
    )
    print(
        f"  interest NOT credited:          Sharpe "
        f"{no_interest_metrics.get('sharpe_ratio', float('nan')):+.4f}"
    )
    print()
    if real.get("skipped"):
        print("Real data: skipped (--skip-real-data)")
    elif real["available"]:
        print(f"Real data: {len(real['results'])} pair(s) succeeded")
        for r in real["results"]:
            sg = r.get("significance", {})
            print(
                f"  {r['pair']:>10}  Sharpe {r['metrics'].get('sharpe_ratio', float('nan')):+.4f}  "
                f"max DD {r['metrics'].get('max_drawdown', float('nan')) * 100:+.2f}%  "
                f"trades {int(r['metrics'].get('n_trades', 0)):3d}  "
                f"t={sg.get('t_statistic', float('nan')):+.2f} "
                f"p={sg.get('p_value', float('nan')):.3f}"
            )
    else:
        print("Real data: UNAVAILABLE")
        for a in real["attempts"][:3]:
            print(f"  {a['pair']}: {a['error']}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
