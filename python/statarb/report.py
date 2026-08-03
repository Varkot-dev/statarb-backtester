"""Charts and tables generated from the engine's output.

Every figure here is drawn from the engine's CSV audit trail, never from a
separate calculation. If a chart disagreed with the metrics file, the chart
would be wrong — there is one source of truth and this module reads it.

Charts are written at a fixed DPI with a fixed style so regenerating them
produces visually identical output, which keeps ``make backtest`` genuinely
reproducible rather than merely re-runnable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Non-interactive backend: the pipeline runs headless in CI, where importing a
# GUI backend either fails or opens a window nobody sees.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: Fixed figure DPI, so regenerated charts are byte-comparable.
FIGURE_DPI = 130

#: A restrained palette. Entries and exits are the two things a reader needs to
#: locate instantly on the z-score chart, so they get the only saturated
#: colours; everything else is neutral.
COLOR_EQUITY = "#1f4e79"
COLOR_DRAWDOWN = "#a4243b"
COLOR_SPREAD = "#3d5a6c"
COLOR_ENTRY = "#1b7f5e"
COLOR_EXIT = "#a4243b"
COLOR_NEUTRAL = "#8c8c8c"
COLOR_BAND = "#d9d9d9"


def _style_axes(ax: plt.Axes) -> None:
    """Apply a consistent, low-chrome axis style.

    Top and right spines carry no information on a time-series chart and
    compete with the data for attention.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_NEUTRAL)
    ax.spines["bottom"].set_color(COLOR_NEUTRAL)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_equity_curve(bars: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot NAV over time with the drawdown profile beneath it.

    The two panels share an x-axis deliberately: the question a reader asks of
    an equity curve is "what did the bad periods cost", and that is only
    answerable when the drawdown is vertically aligned with the curve.
    """
    out_path = Path(out_path)
    nav = bars["nav"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(nav)
    drawdown = nav / peak - 1.0

    fig, (ax_nav, ax_dd) = plt.subplots(
        2,
        1,
        figsize=(11, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1]},
        layout="constrained",
    )

    ax_nav.plot(bars["date"], nav, color=COLOR_EQUITY, linewidth=1.4)
    ax_nav.axhline(
        nav[0], color=COLOR_NEUTRAL, linewidth=0.9, linestyle="--", alpha=0.7
    )
    ax_nav.set_ylabel("NAV ($)")
    ax_nav.set_title(
        "Equity curve and drawdown", loc="left", fontsize=12, fontweight="600"
    )
    _style_axes(ax_nav)

    ax_dd.fill_between(
        bars["date"], drawdown * 100.0, 0.0, color=COLOR_DRAWDOWN, alpha=0.35
    )
    ax_dd.plot(bars["date"], drawdown * 100.0, color=COLOR_DRAWDOWN, linewidth=1.0)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Date")
    _style_axes(ax_dd)

    worst = float(np.min(drawdown))
    ax_dd.annotate(
        f"max drawdown {worst * 100:.2f}%",
        xy=(0.995, 0.08),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color=COLOR_DRAWDOWN,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_spread_and_zscore(
    bars: pd.DataFrame,
    out_path: str | Path,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> Path:
    """Plot the spread and its z-score, with entry and exit fills marked.

    The markers are placed at the bar where the fill occurred, which is one bar
    *after* the z-score that triggered it. That offset is the visual signature
    of next-bar execution: a reader looking closely should see markers sitting
    just past each threshold crossing, never exactly on it. A chart where the
    markers land precisely on the crossing would be evidence of same-bar
    execution.
    """
    out_path = Path(out_path)
    has_signal = bars["zscore"].notna()

    fig, (ax_spread, ax_z) = plt.subplots(
        2,
        1,
        figsize=(11, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.3]},
        layout="constrained",
    )

    ax_spread.plot(
        bars["date"], bars["spread"], color=COLOR_SPREAD, linewidth=1.0
    )
    ax_spread.set_ylabel("Spread (log)")
    ax_spread.set_title(
        "Spread, z-score, and executed trades", loc="left", fontsize=12,
        fontweight="600",
    )
    _style_axes(ax_spread)

    ax_z.plot(bars["date"], bars["zscore"], color=COLOR_SPREAD, linewidth=1.0)
    for level, style in ((entry_threshold, "-"), (exit_threshold, ":")):
        for sign in (1, -1):
            ax_z.axhline(
                sign * level,
                color=COLOR_BAND,
                linewidth=1.0,
                linestyle=style,
                zorder=0,
            )
    ax_z.axhline(0.0, color=COLOR_NEUTRAL, linewidth=0.8, alpha=0.6)

    events = bars["trade_event"].astype(str)
    entries = bars[events.str.startswith("entry")]
    exits = bars[events.str.startswith("exit")]

    if len(entries) > 0:
        ax_z.scatter(
            entries["date"],
            entries["zscore"],
            marker="^",
            s=34,
            color=COLOR_ENTRY,
            zorder=3,
            label=f"entry fill (n={len(entries)})",
        )
    if len(exits) > 0:
        ax_z.scatter(
            exits["date"],
            exits["zscore"],
            marker="v",
            s=34,
            color=COLOR_EXIT,
            zorder=3,
            label=f"exit fill (n={len(exits)})",
        )

    ax_z.set_ylabel("z-score")
    ax_z.set_xlabel("Date")
    if len(entries) > 0 or len(exits) > 0:
        ax_z.legend(loc="upper left", frameon=False, fontsize=9)
    _style_axes(ax_z)

    ax_z.annotate(
        f"entry |z| ≥ {entry_threshold:.1f}, exit |z| ≤ {exit_threshold:.1f}; "
        "fills are next-bar",
        xy=(0.995, 0.03),
        xycoords="axes fraction",
        ha="right",
        fontsize=8,
        color=COLOR_NEUTRAL,
    )

    # Restrict the x-range to bars that actually carry a signal, so the
    # warm-up period does not compress the interesting region.
    if has_signal.any():
        first = bars.loc[has_signal, "date"].iloc[0]
        ax_z.set_xlim(left=first)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_price_series(bars: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot both legs on twin axes, plus the rolling hedge ratio.

    The twin axes are necessary because the legs trade at different price
    levels; plotting them on a shared axis would flatten one into a line. The
    hedge-ratio panel shows how much the estimated relationship moves over the
    sample — a beta that wanders substantially is a warning that the
    cointegrating relationship is unstable.
    """
    out_path = Path(out_path)
    fig, (ax_price, ax_beta) = plt.subplots(
        2,
        1,
        figsize=(11, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1]},
        layout="constrained",
    )

    ax_price.plot(
        bars["date"], bars["price_a"], color=COLOR_EQUITY, linewidth=1.1, label="leg A"
    )
    ax_price.set_ylabel("Price A ($)", color=COLOR_EQUITY)
    ax_price.tick_params(axis="y", labelcolor=COLOR_EQUITY)
    _style_axes(ax_price)

    ax_b = ax_price.twinx()
    ax_b.plot(
        bars["date"], bars["price_b"], color=COLOR_DRAWDOWN, linewidth=1.1, label="leg B"
    )
    ax_b.set_ylabel("Price B ($)", color=COLOR_DRAWDOWN)
    ax_b.tick_params(axis="y", labelcolor=COLOR_DRAWDOWN)
    ax_b.spines["top"].set_visible(False)

    ax_price.set_title(
        "Price legs and rolling hedge ratio", loc="left", fontsize=12,
        fontweight="600",
    )

    ax_beta.plot(bars["date"], bars["hedge_ratio"], color=COLOR_SPREAD, linewidth=1.0)
    ax_beta.set_ylabel("Hedge ratio β")
    ax_beta.set_xlabel("Date")
    _style_axes(ax_beta)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_trade_distribution(trades: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot the distribution of per-trade net PnL and holding periods.

    A pairs strategy's PnL distribution should be left-skewed: many small
    reversion wins against occasional larger stop-loss losses. Seeing that
    shape (or not) says more about the strategy's character than the win rate
    does — a 70% win rate with a fat left tail is not a good strategy.
    """
    out_path = Path(out_path)
    fig, (ax_pnl, ax_hold) = plt.subplots(1, 2, figsize=(11, 4.0), layout="constrained")

    if len(trades) > 0:
        pnl = trades["pnl_net"].to_numpy(dtype=float)
        colors = [COLOR_ENTRY if p > 0 else COLOR_EXIT for p in pnl]
        ax_pnl.bar(range(len(pnl)), pnl, color=colors, width=0.85)
        ax_pnl.axhline(0.0, color=COLOR_NEUTRAL, linewidth=0.9)
        ax_pnl.set_xlabel("Trade number (chronological)")
        ax_pnl.set_ylabel("Net PnL ($)")

        ax_hold.hist(
            trades["holding_bars"].to_numpy(dtype=float),
            bins=min(20, max(5, len(trades) // 2)),
            color=COLOR_SPREAD,
            alpha=0.85,
        )
        ax_hold.set_xlabel("Holding period (bars)")
        ax_hold.set_ylabel("Count")
    else:
        for ax in (ax_pnl, ax_hold):
            ax.text(
                0.5,
                0.5,
                "no trades",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=COLOR_NEUTRAL,
            )

    ax_pnl.set_title("Per-trade net PnL", loc="left", fontsize=11, fontweight="600")
    ax_hold.set_title("Holding periods", loc="left", fontsize=11, fontweight="600")
    _style_axes(ax_pnl)
    _style_axes(ax_hold)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def metrics_markdown_table(metrics: dict[str, float]) -> str:
    """Render the metrics dict as a Markdown table for the README.

    Formatting is per-metric because a Sharpe ratio, a percentage, and a trade
    count each need different precision, and a single format string would make
    at least two of them unreadable.
    """

    def pct(key: str) -> str:
        return f"{metrics.get(key, float('nan')) * 100:.2f}%"

    def num(key: str, places: int = 4) -> str:
        return f"{metrics.get(key, float('nan')):.{places}f}"

    def money(key: str) -> str:
        return f"${metrics.get(key, float('nan')):,.2f}"

    def count(key: str) -> str:
        return f"{int(metrics.get(key, 0)):,}"

    rows = [
        ("**Sharpe ratio** (annualized)", num("sharpe_ratio")),
        ("Sharpe per bar (un-annualized)", num("sharpe_per_bar", 6)),
        ("**Maximum drawdown**", pct("max_drawdown")),
        ("Max drawdown duration", f"{count('max_drawdown_duration')} bars"),
        ("**Number of trades**", count("n_trades")),
        ("Total return", pct("total_return")),
        ("Annualized return", pct("annualized_return")),
        ("Annualized volatility", pct("annualized_volatility")),
        ("Calmar ratio", num("calmar_ratio")),
        ("Win rate", pct("win_rate")),
        ("Average holding period", f"{metrics.get('avg_holding_bars', 0):.1f} bars"),
        ("Average win", money("avg_win")),
        ("Average loss", money("avg_loss")),
        ("Profit factor", num("profit_factor")),
        ("Gross PnL", money("gross_pnl")),
        ("Total transaction costs", money("total_costs")),
        ("Net PnL", money("net_pnl")),
        ("Turnover", f"{metrics.get('turnover', 0):.2f}× initial capital"),
        ("Time in market", pct("exposure_frac")),
        ("Initial NAV", money("initial_nav")),
        ("Final NAV", money("final_nav")),
        ("Bars (total / post-warm-up)", f"{count('n_bars')} / {count('n_trading_bars')}"),
    ]
    lines = ["| Metric | Value |", "| --- | ---: |"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines)


def exit_reason_markdown_table(metrics: dict[str, float]) -> str:
    """Render the breakdown of why positions were closed.

    Worth its own table because the mix is diagnostic: a strategy exiting
    mostly on stop-losses is not the mean-reversion strategy it claims to be,
    whatever its Sharpe says.
    """
    total = max(1.0, float(metrics.get("n_trades", 0)))
    rows = [
        ("Mean reversion (|z| ≤ exit)", "n_reversion"),
        ("Stop-loss (|z| ≥ stop)", "n_stop_loss"),
        ("Max holding period", "n_max_holding"),
        ("End of data (forced close)", "n_end_of_data"),
    ]
    lines = ["| Exit reason | Count | Share |", "| --- | ---: | ---: |"]
    for label, key in rows:
        n = int(metrics.get(key, 0))
        lines.append(f"| {label} | {n} | {n / total * 100:.1f}% |")
    return "\n".join(lines)
