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
        # Not a failure: CI runs offline by design, so the real-data claims
        # cannot be re-derived there. Reporting the count that WAS checked
        # keeps the gap visible instead of letting a smaller number look like
        # a full pass.
        print(
            "note: real data unavailable in this run — the real-data, "
            "variance-ratio, and stability claims were NOT verified. "
            "Run `make backtest` with network access to check all of them."
        )

    # ------------------------------------------------------- variance ratio
    if real.get("available"):
        for r in real["results"]:
            vr = [v for v in r.get("variance_ratio", []) if v["horizon"] == 8]
            if not vr:
                continue
            v = vr[0]
            c.contains(f"{v['variance_ratio']:.3f}", f"VR(8) for {r['pair']}")
            # The README quotes small p-values at 4dp and larger ones at 3dp,
            # which is the right precision for each; accept either rendering.
            c.check(
                f"{v['p_robust']:.4f}" in c.text or f"{v['p_robust']:.3f}" in c.text,
                f"VR p-value for {r['pair']} "
                f"(expected {v['p_robust']:.4f} or {v['p_robust']:.3f})",
            )
        # The README's headline VR claim: XOM/CVX has a finite half-life but is
        # not distinguishable from a random walk.
        xom = next((r for r in real["results"] if r["pair"] == "XOM/CVX"), None)
        if xom:
            v8 = [v for v in xom["variance_ratio"] if v["horizon"] == 8][0]
            c.check(
                not v8["rejects_random_walk"],
                "README claims XOM/CVX is not distinguishable from a random walk",
            )

    # ------------------------------------------------- variance ratio power
    if real.get("available"):
        for r in real["results"]:
            v = r.get("variance_ratio_powered")
            if not v:
                continue
            c.contains(f"**{v['estimated_power']:.2f}**", f"VR power for {r['pair']}")
        # The claim the section rests on: only MA/V is informative.
        informative = [
            r["pair"] for r in real["results"]
            if r.get("variance_ratio_powered", {}).get("is_informative")
        ]
        c.check(
            informative == ["MA/V"],
            f"README claims only MA/V has adequate VR power; got {informative}",
        )
        xom = next((r for r in real["results"] if r["pair"] == "XOM/CVX"), None)
        if xom:
            c.check(
                not xom["variance_ratio_powered"]["is_informative"],
                "README claims the XOM/CVX non-rejection is uninformative",
            )

    # ------------------------------------------------------------- stability
    if real.get("available"):
        for r in real["results"]:
            st = r.get("stability")
            if not st:
                continue
            c.contains(
                f"| {st['first_half_fraction'] * 100:.0f}% | "
                f"{st['second_half_fraction'] * 100:.0f}% |",
                f"stability halves for {r['pair']}",
            )
        # The break-robustness sweep. Its whole point is that the verdict is
        # NOT robust, so the verifier pins the honest reading rather than the
        # flattering one.
        ma_v_rob = next(
            (r["stability"].get("robustness")
             for r in real["results"] if r["pair"] == "MA/V"),
            None,
        )
        if ma_v_rob:
            for row in ma_v_rob["rows"]:
                c.contains(
                    f"| {int(row['window'])} | "
                    f"{row['first_half_fraction'] * 100:.0f}% | "
                    f"{row['second_half_fraction'] * 100:.0f}% | "
                    f"{row['deterioration']:+.3f} |",
                    f"robustness row window={int(row['window'])}",
                )
            c.contains(
                f"{ma_v_rob['fraction_of_windows_firing'] * 100:.0f}% of windows fire",
                "robustness firing fraction",
            )
            c.check(
                ma_v_rob["fraction_of_windows_firing"] < 0.5,
                "README states the break verdict is NOT robust across windows",
            )

        # The headline stability claim: MA/V is the pair that broke.
        ma_v = next((r for r in real["results"] if r["pair"] == "MA/V"), None)
        if ma_v and ma_v.get("stability"):
            st = ma_v["stability"]
            c.check(
                not st["is_stable"],
                "README claims MA/V shows a structural break",
            )
            c.check(
                st["second_half_fraction"] < st["first_half_fraction"],
                "README claims MA/V cointegration deteriorates over the sample",
            )

    # ---------------------------------------------------------- diagnostics
    if "diagnostics" in summary:
        d = summary["diagnostics"]
        for r in d["half_life_sensitivity"]:
            c.contains(
                f"| {int(r['half_life'])} | {r['sharpe']:+.3f} |",
                f"half-life sensitivity row hl={r['half_life']}",
            )
        for r in d["window_ratio_sensitivity"]:
            c.contains(
                f"| {r['zscore_window']} | {r['window_ratio']:.1f} | {r['sharpe']:+.3f} |",
                f"window-ratio row w={r['zscore_window']}",
            )
        ds = d["deflated_sharpe"]
        c.contains(f"{ds['observed_sharpe']:+.3f}", "deflated: observed Sharpe")
        c.contains(
            f"{ds['expected_max_under_null']:+.3f}", "deflated: luck threshold"
        )
        c.contains(
            f"{ds['deflated_sharpe_statistic']:+.3f}".replace("-", "−"),
            "deflated: statistic",
        )
        c.check(
            not ds["survives"],
            "README claims the searched result does NOT survive deflation",
        )
        wfs = d.get("walk_forward_summary")
        if wfs and wfs.get("n_kept", 0) > 0:
            c.contains(
                f"{wfs['mean_oos_sharpe']:+.3f}".replace("-", "−"),
                "walk-forward mean OOS Sharpe",
            )
            # The mean must never appear without its standard error. Reporting a
            # 2-pair average as a point estimate is the error this guards.
            c.contains(
                f"{wfs['se_mean_oos_sharpe']:.3f}",
                "walk-forward SE must be reported alongside the mean",
            )
            c.check(
                f"{wfs['n_dropped']} of {wfs['n_kept'] + wfs['n_dropped']}" in c.text,
                "README must state how many pairs the drop rule discarded",
            )

    # ------------------------------------------------------- kalman RMSE
    if "kalman_rmse" in summary:
        r = summary["kalman_rmse"]
        for key in ("min", "p25", "median", "p75", "max"):
            c.contains(f"{r[key]:.1f}%", f"kalman RMSE {key}")
        c.contains(f"{int(r['n_seeds'])} seeds", "kalman RMSE seed count")
        # The README must quote the MEDIAN, not the maximum. An earlier draft
        # quoted a single near-best seed; this pins the honest statistic.
        c.check(
            f"median\n{r['median']:.1f}%" in c.text
            or f"median**\n{r['median']:.1f}%" in c.text
            or f"{r['median']:.1f}% lower RMSE" in c.text,
            f"README must quote the MEDIAN RMSE improvement ({r['median']:.1f}%)",
        )

    # -------------------------------------------------- kalman memory sweep
    if "kalman_memory_sweep" in summary:
        ks = summary["kalman_memory_sweep"]["rows"]
        for r in ks:
            c.contains(
                f"| {r['memory_ratio']:.0f} | {r['effective_memory_bars']:.0f} | "
                f"{r['sharpe_ratio']:+.3f} |",
                f"kalman sweep row ratio={r['memory_ratio']}",
            )
        # The claim is monotone improvement with memory, not a specific level.
        c.check(
            all(a["sharpe_ratio"] <= b["sharpe_ratio"] + 1e-9
                for a, b in zip(ks, ks[1:])),
            "README claims Sharpe rises monotonically with Kalman memory",
        )
        # And every row must represent a real backtest, not an empty trade list.
        # The first version of this table reported 0-10 trades; the numbers were
        # artifacts. This pins that it cannot recur.
        c.check(
            all(r["n_trades"] >= 20 for r in ks),
            "kalman sweep rows must have enough trades to be meaningful "
            f"(min {min(r['n_trades'] for r in ks)})",
        )

    # ------------------------------------------------------ leakage calibration
    if "leakage_calibration" in summary:
        rows = summary["leakage_calibration"]["rows"]
        shift = [r for r in rows if r["leak_type"] == "timing_shift"]
        filt = [r for r in rows if r["leak_type"] == "outcome_filter"]

        c.contains(f"{shift[0]['sharpe_ratio']:+.2f}", "calibration: honest Sharpe (shift curve)")
        c.contains(
            f"{shift[6]['sharpe_ratio']:+.2f}".replace("-", "−"),
            "calibration: timing-shift Sharpe at k=6",
        )
        c.contains(f"{filt[-1]['sharpe_ratio']:+.2f}", "calibration: fully-filtered Sharpe")
        c.contains(
            f"{shift[0]['win_rate'] * 100:.0f}%", "calibration: honest win rate"
        )
        c.contains(
            f"{shift[6]['win_rate'] * 100:.0f}%", "calibration: shifted win rate"
        )
        c.contains(
            f"{int(filt[0]['n_trades'])} → {int(filt[-1]['n_trades'])}",
            "calibration: trade-count drop under filtering",
        )
        # The direction claims must match the measurements, not just the text.
        c.check(
            shift[6]["sharpe_ratio"] < shift[0]["sharpe_ratio"],
            "README claims timing shift DEGRADES performance",
        )
        c.check(
            filt[-1]["sharpe_ratio"] > filt[0]["sharpe_ratio"],
            "README claims outcome filtering INFLATES Sharpe",
        )
        c.check(
            filt[-1]["n_trades"] < filt[0]["n_trades"],
            "README claims outcome filtering reduces the trade count",
        )
        c.check(
            filt[-1]["win_rate"] > filt[0]["win_rate"],
            "README claims outcome filtering raises the win rate",
        )

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
    scope = "all" if real.get("available") else "offline-only"
    print(
        f"{c.passed} README claims verified against reports/summary.json "
        f"({scope})"
    )
    if c.failures:
        print(f"\n{len(c.failures)} MISMATCH(ES):", file=sys.stderr)
        for failure in c.failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("All README numbers match the generated artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
