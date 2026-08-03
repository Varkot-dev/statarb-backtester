#!/usr/bin/env python3
"""Verify that every number in the README matches the generated artifacts.

The README makes a strong claim — that nothing in it is hand-typed and every
figure comes from ``make backtest``. This script is what makes that claim
checkable rather than merely stated. It re-reads ``reports/summary.json`` and
the CSVs, formats each value exactly as the README does, and asserts the
resulting string appears in the file.

Run via ``make verify`` (and in CI). It exits non-zero on any mismatch, so a
result that drifts from its documentation fails the build instead of quietly
becoming a lie.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import pandas as pd  # noqa: E402

README = REPO_ROOT / "README.md"
SUMMARY = REPO_ROOT / "reports" / "summary.json"


class Checker:
    """Collects pass/fail results so every claim is reported, not just the first."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.passed = 0
        self.failures: list[str] = []

    def contains(self, needle: str, label: str) -> None:
        if needle in self.text:
            self.passed += 1
        else:
            self.failures.append(f"{label}: expected to find {needle!r}")

    def check(self, condition: bool, label: str) -> None:
        if condition:
            self.passed += 1
        else:
            self.failures.append(label)


def main() -> int:
    if not SUMMARY.exists():
        print(f"error: {SUMMARY} not found — run 'make backtest' first", file=sys.stderr)
        return 1

    summary = json.loads(SUMMARY.read_text())
    c = Checker(README.read_text())

    by_key = {d["key"]: d for d in summary["synthetic"]}

    # ---------------------------------------------------------------- primary
    primary = by_key["coint_medium"]
    m, sig, coint = primary["metrics"], primary["significance"], primary["cointegration"]

    c.contains(f"{m['sharpe_ratio']:+.4f}".replace("+", "+"), "primary Sharpe (4dp)")
    c.contains(f"{m['sharpe_ratio']:.2f}", "primary Sharpe (2dp)")
    c.contains(f"{m['max_drawdown'] * 100:.2f}%".replace("-", "−"), "primary max drawdown")
    c.contains(f"| **{int(m['n_trades'])}** |", "primary trade count")
    c.contains(f"{sig['t_statistic']:.2f}", "primary t-statistic")
    c.contains(f"{sig['p_value']:.4f}", "primary p-value")
    c.contains(f"{m['total_return'] * 100:.2f}%", "primary total return")
    c.contains(f"{m['annualized_return'] * 100:.2f}%", "primary annualized return")
    c.contains(f"{m['win_rate'] * 100:.1f}%", "primary win rate")
    c.contains(f"{m['calmar_ratio']:.2f}", "primary Calmar")
    c.contains(f"{m['profit_factor']:.2f}", "primary profit factor")
    c.contains(f"${m['total_costs']:,.2f}", "primary total costs")
    c.contains(f"+${m['total_interest']:,.0f}", "primary interest")
    c.contains(f"+${m['net_pnl'] - m['total_interest']:,.0f}", "primary trading PnL")
    c.contains(f"{m['avg_holding_bars']:.1f} bars", "primary holding period")
    c.contains(f"{m['turnover']:.1f}×", "primary turnover")
    c.contains(f"{int(m['max_drawdown_duration'])} bars", "primary drawdown duration")
    c.contains(f"{sig['sharpe_ci_low']:+.2f}, {sig['sharpe_ci_high']:+.2f}", "primary CI")
    c.contains(f"{coint['hedge_ratio']:.3f}", "primary estimated beta")
    c.contains(f"{coint['half_life']:.1f} bars", "primary estimated half-life")
    c.contains(f"{int(m['n_reversion'])} reversion", "primary exit breakdown")

    # ------------------------------------------------------- negative control
    neg = by_key["independent"]
    nm, nsig, ncoint = neg["metrics"], neg["significance"], neg["cointegration"]
    trading = nm["net_pnl"] - nm["total_interest"]

    c.contains(f"+{trading / 100_000 * 100:.1f}%", "negative-control trading return")
    c.contains(f"+${trading:,.0f}", "negative-control trading PnL")
    c.contains(f"{nsig['t_statistic']:.2f}", "negative-control t-statistic")
    c.contains(f"{nsig['p_value']:.3f}", "negative-control p-value")
    c.contains(f"{nm['max_drawdown'] * 100:.2f}%".replace("-", "−"), "negative-control max DD")
    c.contains(f"{ncoint['eg_pvalue']:.3f}", "negative-control EG p-value")
    c.contains(f"{ncoint['johansen_trace_stat']:.2f}", "negative-control Johansen")
    c.contains(f"{ncoint['half_life']:.0f} bars", "negative-control half-life")
    c.check(
        not nsig["is_significant"],
        "negative control must NOT be statistically significant",
    )

    # ------------------------------------------------------------ slow dataset
    slow = by_key["coint_slow"]
    c.contains(f"{slow['metrics']['sharpe_ratio']:.2f}", "slow Sharpe")
    c.contains(f"{slow['significance']['t_statistic']:.2f}", "slow t-statistic")
    c.contains(f"{slow['significance']['p_value']:.3f}", "slow p-value")

    # -------------------------------------------------- cash-interest finding
    comparison = summary["cash_interest_comparison"]
    c.contains(
        f"{comparison['without_interest']['sharpe_ratio']:.3f}".replace("-", "−"),
        "no-interest Sharpe",
    )
    c.contains(f"{comparison['with_interest']['sharpe_ratio']:.3f}", "with-interest Sharpe")
    c.contains(
        f"{comparison['without_interest']['total_return'] * 100:.2f}%",
        "no-interest total return",
    )

    # ------------------------------------------------------------- real data
    real = summary["real_data"]
    if real.get("available"):
        n_significant = sum(
            1 for r in real["results"] if r["significance"]["is_significant"]
        )
        c.check(
            n_significant == 0,
            f"README claims 0 of 5 real pairs are significant; found {n_significant}",
        )
        c.contains(f"**0 of {len(real['results'])} significant**", "real-data verdict")

        for r in real["results"]:
            ticker = r["pair"].replace("/", "/")
            rm, rs, rc = r["metrics"], r["significance"], r["cointegration"]
            c.contains(ticker, f"real pair {ticker} listed")
            c.contains(
                f"{rm['sharpe_ratio']:+.3f}".replace("-", "−"),
                f"real {ticker} Sharpe",
            )
            c.contains(
                f"{rc['correlation_levels']:.3f}", f"real {ticker} level correlation"
            )
            c.contains(f"{rs['p_value']:.3f}", f"real {ticker} p-value")

        cointegrated = [r for r in real["results"] if r["cointegration"]["is_cointegrated"]]
        c.check(
            len(cointegrated) == 1 and cointegrated[0]["pair"] == "MA/V",
            "README claims MA/V is the only cointegrated real pair",
        )
    else:
        print("note: real data unavailable in this run; skipping those checks")

    # ------------------------------------------------------------------ sweep
    sweep = pd.read_csv(REPO_ROOT / summary["sweep_csv"])
    c.contains(f"{len(sweep)} combinations", "sweep combination count")
    c.contains(f"{sweep.sharpe_ratio.min():.3f}".replace("-", "−"), "sweep min Sharpe")
    c.contains(f"{sweep.sharpe_ratio.max():.3f}", "sweep max Sharpe")
    c.contains(f"{sweep.sharpe_ratio.median():.3f}", "sweep median Sharpe")
    c.contains(f"{(sweep.sharpe_ratio > 0).mean() * 100:.1f}%", "sweep positive fraction")

    pivot = sweep.pivot_table(
        index="window", columns="entry", values="sharpe_ratio", aggfunc="mean"
    )
    for window in pivot.index:
        for entry in pivot.columns:
            value = pivot.loc[window, entry]
            c.contains(
                f"{value:+.3f}".replace("-", "−"),
                f"sweep cell window={window} entry={entry}",
            )

    # ------------------------------------------------------------------ report
    print(f"{c.passed} README claims verified against reports/summary.json")
    if c.failures:
        print(f"\n{len(c.failures)} MISMATCH(ES):", file=sys.stderr)
        for failure in c.failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("All README numbers match the generated artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
