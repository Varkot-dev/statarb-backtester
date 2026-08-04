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


def plot_equity_decomposition(bars: pd.DataFrame, out_path: str | Path) -> Path:
    """Separate the equity curve into trading PnL and interest on idle cash.

    Why this chart exists: on a strategy that holds a position only ~24% of
    bars, accrued interest dominates the NAV path. A plain equity curve then
    looks like a near-straight line and flatters the strategy badly — the
    smoothness is the risk-free rate, not the trading.

    Plotting cumulative trading PnL on its own shows what the strategy actually
    did. It is far choppier, and that is the honest picture.

    ``trading_pnl[t] = nav[t] - nav[0] - cumulative_interest[t]``, which follows
    directly from the accounting identity in :mod:`statarb.io`.
    """
    out_path = Path(out_path)
    nav = bars["nav"].to_numpy(dtype=float)
    cumulative_interest = np.cumsum(bars["interest_this_bar"].to_numpy(dtype=float))
    total_gain = nav - nav[0]
    trading_pnl = total_gain - cumulative_interest

    fig, (ax_split, ax_trading) = plt.subplots(
        2,
        1,
        figsize=(11, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1]},
        layout="constrained",
    )

    ax_split.plot(
        bars["date"], total_gain, color=COLOR_EQUITY, linewidth=1.5, label="total gain"
    )
    ax_split.plot(
        bars["date"],
        cumulative_interest,
        color=COLOR_NEUTRAL,
        linewidth=1.2,
        linestyle="--",
        label="interest on idle cash",
    )
    ax_split.plot(
        bars["date"],
        trading_pnl,
        color=COLOR_ENTRY,
        linewidth=1.4,
        label="trading PnL",
    )
    ax_split.axhline(0.0, color=COLOR_NEUTRAL, linewidth=0.8, alpha=0.6)
    ax_split.set_ylabel("Cumulative $")
    ax_split.set_title(
        "What actually drove the return", loc="left", fontsize=12, fontweight="600"
    )
    ax_split.legend(loc="upper left", frameon=False, fontsize=9)
    _style_axes(ax_split)

    share = (
        trading_pnl[-1] / total_gain[-1] * 100.0 if abs(total_gain[-1]) > 1e-9 else 0.0
    )
    ax_split.annotate(
        f"trading is {share:.0f}% of the total gain",
        xy=(0.995, 0.06),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color=COLOR_ENTRY,
    )

    # Trading PnL alone, on its own scale — the strategy without the tailwind.
    ax_trading.plot(bars["date"], trading_pnl, color=COLOR_ENTRY, linewidth=1.2)
    ax_trading.fill_between(
        bars["date"], trading_pnl, 0.0, color=COLOR_ENTRY, alpha=0.15
    )
    ax_trading.axhline(0.0, color=COLOR_NEUTRAL, linewidth=0.8)
    ax_trading.set_ylabel("Trading PnL ($)")
    ax_trading.set_xlabel("Date")
    ax_trading.set_title(
        "Trading PnL alone (note the scale — this is the real path)",
        loc="left",
        fontsize=10,
        color=COLOR_NEUTRAL,
    )
    _style_axes(ax_trading)

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


def plot_leakage_calibration(calibration: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot the two leakage dose-response curves side by side.

    The point of showing them together is the **asymmetry**: they run in
    opposite directions. Timing-shift leakage (an off-by-one in a join) destroys
    a mean-reversion strategy; outcome-filter leakage (a ``shift(-1)`` on a
    feature) inflates it. Two different bugs, two different signatures.

    The honest result is marked on both panels, because the reader's first
    question is "where does the real backtest sit on this curve?"
    """
    out_path = Path(out_path)
    shift = calibration[calibration["leak_type"] == "timing_shift"].sort_values("dose")
    filt = calibration[calibration["leak_type"] == "outcome_filter"].sort_values("dose")

    fig, (ax_shift, ax_filter) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), layout="constrained"
    )

    # --- Timing shift: harmful ---
    ax_shift.plot(
        shift["dose"], shift["sharpe_ratio"],
        color=COLOR_DRAWDOWN, linewidth=1.8, marker="o", markersize=4,
    )
    ax_shift.axhline(0.0, color=COLOR_NEUTRAL, linewidth=0.9)
    honest_shift = float(shift.iloc[0]["sharpe_ratio"])
    ax_shift.scatter(
        [0], [honest_shift], s=90, facecolor="white",
        edgecolor=COLOR_EQUITY, linewidth=2, zorder=5,
    )
    ax_shift.annotate(
        f"honest\n{honest_shift:+.2f}",
        xy=(0, honest_shift), xytext=(1.4, honest_shift - 0.95),
        fontsize=9, color=COLOR_EQUITY,
        arrowprops={"arrowstyle": "-", "color": COLOR_EQUITY, "linewidth": 0.9},
    )
    ax_shift.set_xlabel("bars of timing shift (k)")
    ax_shift.set_ylabel("Sharpe ratio")
    ax_shift.set_title(
        "Timing shift DESTROYS performance", loc="left", fontsize=11,
        fontweight="600", color=COLOR_DRAWDOWN,
    )
    ax_shift.annotate(
        "off-by-one in a resample or join;\nenters before the extreme it wants",
        xy=(0.97, 0.72), xycoords="axes fraction", ha="right", va="top",
        fontsize=8, color=COLOR_NEUTRAL,
    )
    _style_axes(ax_shift)

    # --- Outcome filter: helpful (to the fraudster) ---
    ax_filter.plot(
        filt["dose"] * 100.0, filt["sharpe_ratio"],
        color=COLOR_ENTRY, linewidth=1.8, marker="o", markersize=4,
    )
    honest_filter = float(filt.iloc[0]["sharpe_ratio"])
    ax_filter.scatter(
        [0], [honest_filter], s=90, facecolor="white",
        edgecolor=COLOR_EQUITY, linewidth=2, zorder=5,
    )
    ax_filter.annotate(
        f"honest {honest_filter:+.2f}",
        xy=(0, honest_filter), xytext=(9, honest_filter - 0.008),
        fontsize=9, color=COLOR_EQUITY,
        arrowprops={"arrowstyle": "-", "color": COLOR_EQUITY, "linewidth": 0.9},
    )
    ax_filter.set_xlabel("% of losing trades skipped")
    ax_filter.set_ylabel("Sharpe ratio")
    ax_filter.set_title(
        "Outcome filtering INFLATES it", loc="left", fontsize=11,
        fontweight="600", color=COLOR_ENTRY,
    )
    ax_filter.annotate(
        "shift(-1) on a feature, or same-bar fills;\nfewer trades, higher win rate",
        xy=(0.97, 0.20), xycoords="axes fraction", ha="right", va="top",
        fontsize=8, color=COLOR_NEUTRAL,
    )
    _style_axes(ax_filter)

    fig.suptitle(
        "What a known dose of lookahead bias does to the reported result",
        fontsize=12.5, fontweight="600", x=0.01, ha="left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_honesty_cascade(
    stages: list[tuple[str, str, float, str]],
    out_path: str | Path,
    deflated_note: str = "",
) -> Path:
    """Plot the sequence of results that each looked like alpha and then wasn't.

    Why a cascade rather than a bar chart: the argument is *ordinal*. Each row
    is a more demanding test than the one above it, and the claim is that the
    apparent edge shrinks monotonically as the test gets more honest. A bar
    chart invites the reader to compare heights in any order; a descending
    cascade with a connecting spine forces the sequence, which is the point.

    Bars grow rightward from a common zero so the eye reads magnitude, and each
    stage is annotated in place — the reader should never have to consult a
    legend to learn which test knocked which result down.

    Colour carries no information the shape does not: surviving stages are drawn
    solid and knocked-down stages hatched, so the figure is legible in greyscale
    and to a colour-blind reader.

    Args:
        stages: ``(label, detail, value, verdict)`` per row, top to bottom, in
            the order the tests were applied. ``verdict`` is ``"alpha"`` for a
            result that looked like a discovery or ``"killed"`` for one that
            did not survive.
        out_path: Destination PNG.
        deflated_note: Text placed against the deflated-Sharpe row. The near
            equality of observed and luck-expected Sharpe is the crux of the
            whole figure and is too important to leave to the axis.
    """
    out_path = Path(out_path)

    fig, ax = plt.subplots(figsize=(11.5, 5.6), layout="constrained")

    n = len(stages)
    y_positions = list(range(n - 1, -1, -1))
    values = [v for _, _, v, _ in stages]
    span = max(abs(min(values)), abs(max(values)), 0.1)

    for y, (label, detail, value, verdict) in zip(y_positions, stages):
        killed = verdict != "alpha"
        # Hatching, not hue, is what survives a greyscale print.
        ax.barh(
            y,
            value,
            height=0.46,
            color=COLOR_DRAWDOWN if killed else COLOR_EQUITY,
            alpha=0.30 if killed else 0.85,
            hatch="///" if killed else None,
            edgecolor=COLOR_DRAWDOWN if killed else COLOR_EQUITY,
            linewidth=1.1,
            zorder=3,
        )
        # The value sits on the far side of the bar tip, so it never collides
        # with the bar however long the bar is.
        ax.annotate(
            f"{value:+.3f}",
            xy=(value, y),
            xytext=(6 if value >= 0 else -6, 0),
            textcoords="offset points",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
            fontweight="600",
            color=COLOR_DRAWDOWN if killed else COLOR_EQUITY,
            zorder=4,
        )
        # Row label to the left of the zero line, in the negative gutter.
        ax.annotate(
            label,
            xy=(0, y + 0.16),
            xytext=(-14, 0),
            textcoords="offset points",
            va="center",
            ha="right",
            fontsize=10,
            fontweight="600",
            color="#2b2b2b",
        )
        ax.annotate(
            detail,
            xy=(0, y - 0.20),
            xytext=(-14, 0),
            textcoords="offset points",
            va="center",
            ha="right",
            fontsize=8.2,
            color=COLOR_NEUTRAL,
        )

    # The connecting spine: an explicit downward arrow between consecutive
    # stages, so the reader sees a sequence of knock-downs rather than four
    # independent measurements that happen to be stacked.
    spine_x = span * 0.045
    for upper, lower in zip(y_positions, y_positions[1:]):
        ax.annotate(
            "",
            xy=(spine_x, lower + 0.26),
            xytext=(spine_x, upper - 0.26),
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLOR_NEUTRAL,
                "linewidth": 1.1,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=2,
        )

    if deflated_note:
        # Anchored to the deflated row (index 2 from the top by construction of
        # the caller's stage list) if it exists, otherwise to the last row. It
        # sits in the gap *below* the row rather than on top of the bar, so it
        # covers neither the hatching nor the value label.
        note_y = (y_positions[2] if n > 2 else y_positions[-1]) - 0.52
        ax.annotate(
            deflated_note,
            xy=(span * 0.02, note_y),
            fontsize=8.6,
            color=COLOR_DRAWDOWN,
            va="center",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.34",
                "facecolor": "white",
                "edgecolor": COLOR_DRAWDOWN,
                "linewidth": 0.8,
                "alpha": 0.96,
            },
            zorder=6,
        )

    ax.axvline(0.0, color=COLOR_NEUTRAL, linewidth=1.0, zorder=1)
    ax.set_yticks([])
    ax.set_ylim(-0.75, n - 0.25)
    # Only as much negative room as the data actually needs. A symmetric range
    # would spend half the canvas on an empty region and shrink every bar.
    left = min(min(values), 0.0)
    ax.set_xlim(left - span * 0.08 if left < 0 else -span * 0.03, span * 1.30)
    ax.set_xlabel("Sharpe ratio")
    ax.spines["left"].set_visible(False)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)

    # The subtitle occupies the line above the axes, so the title is lifted
    # clear of it rather than being overprinted.
    ax.annotate(
        "each row is a stricter test than the one above it",
        xy=(0.0, 1.015),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=9,
        color=COLOR_NEUTRAL,
    )
    ax.set_title(
        "Every apparent edge, and the test that removed it",
        loc="left",
        fontsize=12.5,
        fontweight="600",
        pad=26,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_memory_law(
    ratio_rows: pd.DataFrame,
    kalman_rows: pd.DataFrame,
    out_path: str | Path,
    min_ratio: float = 5.0,
) -> Path:
    """Plot both estimators against a shared memory-to-half-life axis.

    Two estimators with nothing in common structurally — a rolling OLS window
    with a hard cutoff, and a Kalman filter with no window at all, weighting the
    past by exponential decay — are placed on one x-axis because their *memory*
    is the quantity that turns out to matter. For the window that is the window
    length; for the filter it is ``1/sqrt(Q/R)``. Both divided by the spread's
    half-life.

    The chart makes two claims at once, and needs both to be honest:

    - **Both degrade.** The windowless estimator is not exempt, so the constraint
      is a property of the estimation problem rather than an artifact of
      windowing.
    - **The windowed one degrades harder.** A hard cutoff discards the 61st
      observation entirely while weighting the 60th fully; exponential decay has
      no such discontinuity, and the gap between the two curves is the price of
      that discontinuity.

    The two series are distinguished by linestyle and marker as well as hue, so
    the comparison survives a greyscale print — which is the medium a chart in a
    README is most often read in.
    """
    out_path = Path(out_path)

    rolling = ratio_rows.sort_values("window_ratio")
    kalman = kalman_rows.sort_values("memory_ratio")

    fig, ax = plt.subplots(figsize=(11.0, 5.4), layout="constrained")

    # The danger zone: memory shorter than ~5 half-lives. Shaded rather than
    # merely lined, because the claim is about a region, not a threshold value.
    ax.axvspan(0, min_ratio, color=COLOR_DRAWDOWN, alpha=0.07, zorder=0)
    ax.axvline(min_ratio, color=COLOR_DRAWDOWN, linewidth=1.0, linestyle=":", zorder=1)
    ax.axhline(0.0, color=COLOR_NEUTRAL, linewidth=0.9, zorder=1)

    ax.plot(
        rolling["window_ratio"],
        rolling["sharpe"],
        color=COLOR_DRAWDOWN,
        linewidth=1.9,
        linestyle="-",
        marker="o",
        markersize=6,
        zorder=4,
    )
    ax.plot(
        kalman["memory_ratio"],
        kalman["sharpe_ratio"],
        color=COLOR_EQUITY,
        linewidth=1.7,
        linestyle="--",
        marker="s",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.4,
        zorder=4,
    )

    # Label the series on the curves themselves. A legend would put the two
    # names in a corner and force the reader to map colours back to lines.
    # Both labels are anchored inside the axes and right-aligned at their
    # curve's endpoint, so neither can run off the right edge however far the
    # sweep extends.
    r_last = rolling.iloc[-1]
    ax.annotate(
        "rolling OLS\nhard cutoff",
        xy=(float(r_last["window_ratio"]), float(r_last["sharpe"])),
        xytext=(-10, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9.5,
        fontweight="600",
        color=COLOR_DRAWDOWN,
    )
    # The two curves converge at the right-hand end and the region between them
    # is narrow throughout, so the Kalman label is parked in the genuinely empty
    # lower-middle of the plot and tied to its curve with a leader line.
    k_first = kalman.iloc[0]
    ax.annotate(
        "Kalman filter\nexponential decay, no window",
        xy=(float(k_first["memory_ratio"]), float(k_first["sharpe_ratio"])),
        xytext=(0.30, 0.20),
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="600",
        color=COLOR_EQUITY,
        arrowprops={
            "arrowstyle": "-",
            "color": COLOR_EQUITY,
            "linewidth": 0.9,
            "alpha": 0.7,
        },
    )

    # The quantitative core: how far each estimator falls across the sweep.
    r_lo, r_hi = float(rolling["sharpe"].iloc[0]), float(rolling["sharpe"].max())
    k_lo, k_hi = float(kalman["sharpe_ratio"].iloc[0]), float(
        kalman["sharpe_ratio"].max()
    )
    ax.annotate(
        f"rolling OLS falls {r_hi:+.2f} → {r_lo:+.2f}\n"
        f"Kalman falls {k_hi:+.2f} → {k_lo:+.2f}\n"
        "the discontinuity costs the difference",
        xy=(0.985, 0.06),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8.6,
        color=COLOR_NEUTRAL,
    )
    ax.annotate(
        f"memory < {min_ratio:.0f} × half-life:\nthe trailing location estimate is\n"
        "contaminated by the deviation\nit is trying to measure",
        xy=(0.03, 0.955),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8.6,
        color=COLOR_DRAWDOWN,
    )

    ax.set_xlabel("estimator memory ÷ spread half-life")
    ax.set_ylabel("Sharpe ratio")
    # A little headroom on the right so the on-curve labels have somewhere to
    # sit without being clipped by the axes edge.
    x_max = max(
        float(rolling["window_ratio"].max()), float(kalman["memory_ratio"].max())
    )
    ax.set_xlim(0, x_max * 1.06)
    ax.annotate(
        "both degrade as memory approaches the half-life — the windowless "
        "filter does not escape it",
        xy=(0.0, 1.015),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=9,
        color=COLOR_NEUTRAL,
    )
    ax.set_title(
        "One law, two estimators: it is the memory, not the window",
        loc="left",
        fontsize=12.5,
        fontweight="600",
        pad=26,
    )
    _style_axes(ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_leakage_signature(calibration: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot the fingerprint of each leak type as a practitioner's lookup.

    :func:`plot_leakage_calibration` answers "what does a dose of leakage do to
    Sharpe?". This chart answers the question a practitioner actually has, which
    runs the other way: *my backtest looks like this — which bug do I have?*

    That inversion drives the layout. Three rows, one per observable a reader
    can read off their own backtest report — Sharpe, win rate, and trade count —
    each plotted against the leak dose for both leak types on a shared row. The
    diagnostic content is in the **pattern of signs across the three rows**, not
    in any single curve:

    - **Timing shift**: Sharpe down, win rate down, trade count flat. The rule
      still fires the same number of times; it just fires early and holds the
      position while the spread travels *into* the extreme.
    - **Outcome filter**: Sharpe up, win rate up, trade count **down**. That
      third sign is the tell, and it is the one a reader would not think to
      check — the cheat declines trades rather than improving them.

    Dose is normalised to 0–100% of each sweep's own range so the two leak types
    share an x-axis despite being measured in different units (bars versus
    percent of losers skipped). The absolute doses are annotated on the top row
    so nothing is hidden by the normalisation.
    """
    out_path = Path(out_path)

    shift = calibration[calibration["leak_type"] == "timing_shift"].sort_values("dose")
    filt = calibration[calibration["leak_type"] == "outcome_filter"].sort_values("dose")

    def normalised(frame: pd.DataFrame) -> np.ndarray:
        dose = frame["dose"].to_numpy(dtype=float)
        top = dose.max()
        return dose / top * 100.0 if top > 0 else dose

    x_shift = normalised(shift)
    x_filt = normalised(filt)

    rows = (
        ("sharpe_ratio", "Sharpe ratio", 1.0),
        ("win_rate", "Win rate (%)", 100.0),
        ("n_trades", "Trade count", 1.0),
    )

    fig, axes = plt.subplots(
        3, 1, figsize=(11.0, 8.2), sharex=True, layout="constrained"
    )

    for ax, (column, ylabel, scale) in zip(axes, rows):
        ax.plot(
            x_shift,
            shift[column].to_numpy(dtype=float) * scale,
            color=COLOR_DRAWDOWN,
            linewidth=1.8,
            linestyle="-",
            marker="o",
            markersize=4.5,
            zorder=3,
        )
        ax.plot(
            x_filt,
            filt[column].to_numpy(dtype=float) * scale,
            color=COLOR_ENTRY,
            linewidth=1.8,
            linestyle="--",
            marker="s",
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.3,
            zorder=3,
        )
        ax.set_ylabel(ylabel)
        _style_axes(ax)

        # Direction-of-travel summary in a clear margin to the right of the
        # data. These, not the curves, are what a reader matches their own
        # backtest against — so they get reserved space rather than being laid
        # over a line, where a curve ending high would obscure them.
        ax.set_xlim(-3.0, 100.0 + 34.0)
        ax.axvline(103.0, color=COLOR_BAND, linewidth=0.9, zorder=1)
        # Ticks stop at the data. Letting them run into the summary margin
        # would imply doses beyond 100% of the sweep, which do not exist.
        ax.set_xticks([0, 20, 40, 60, 80, 100])

        shift_delta = float(shift[column].iloc[-1] - shift[column].iloc[0])
        filt_delta = float(filt[column].iloc[-1] - filt[column].iloc[0])
        for delta, colour, frac in (
            (shift_delta, COLOR_DRAWDOWN, 0.72),
            (filt_delta, COLOR_ENTRY, 0.34),
        ):
            if abs(delta) < 1e-9:
                glyph, label_colour = "→ flat", COLOR_NEUTRAL
            elif delta > 0:
                glyph, label_colour = "▲ up", colour
            else:
                glyph, label_colour = "▼ down", colour
            ax.annotate(
                glyph,
                xy=(112.0, frac),
                xycoords=ax.get_xaxis_transform(),
                ha="left",
                va="center",
                fontsize=9.5,
                fontweight="600",
                color=label_colour,
            )

    axes[0].annotate(
        "two leaks, opposite directions — and the trade count is what tells "
        "them apart",
        xy=(0.0, 1.03),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=9,
        color=COLOR_NEUTRAL,
    )
    axes[0].set_title(
        "The diagnostic signature: read your own backtest off these three rows",
        loc="left",
        fontsize=12.5,
        fontweight="600",
        pad=28,
    )

    # Name the two series on the top panel, placed in the open region each
    # curve leaves behind: the timing-shift curve dives, so its label goes high
    # on the left; the outcome-filter curve stays flat and high, so its label
    # goes just beneath it.
    axes[0].annotate(
        f"timing shift  (0 → {int(shift['dose'].max())} bars)",
        xy=(0.30, 0.50),
        xycoords="axes fraction",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="600",
        color=COLOR_DRAWDOWN,
    )
    axes[0].annotate(
        f"outcome filter  (0 → {filt['dose'].max() * 100:.0f}% of losers skipped)",
        xy=(0.30, 0.80),
        xycoords="axes fraction",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="600",
        color=COLOR_ENTRY,
    )
    # Header for the summary margin, so the arrow glyphs are self-explaining.
    axes[0].annotate(
        "direction\nof travel",
        xy=(112.0, 1.04),
        xycoords=axes[0].get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=8.2,
        color=COLOR_NEUTRAL,
    )

    # The three-part fingerprint, stated explicitly. This is the payload of the
    # whole chart: a reader should be able to leave with these two rules.
    n_filt_lo = int(filt["n_trades"].iloc[0])
    n_filt_hi = int(filt["n_trades"].iloc[-1])
    wr_lo = float(filt["win_rate"].iloc[0]) * 100.0
    wr_hi = float(filt["win_rate"].iloc[-1]) * 100.0
    axes[2].annotate(
        f"OUTCOME FILTERING — the three-part fingerprint:\n"
        f"Sharpe ▲, win rate ▲ ({wr_lo:.0f}% → {wr_hi:.0f}%), "
        f"trade count ▼ ({n_filt_lo} → {n_filt_hi}).\n"
        "Fewer trades with a higher win rate means the cheat is *declining* "
        "trades,\nnot improving them. Check trade count against signal count.",
        xy=(0.015, 0.30),
        xycoords="axes fraction",
        ha="left",
        va="center",
        fontsize=8.6,
        color=COLOR_ENTRY,
        bbox={
            "boxstyle": "round,pad=0.40",
            "facecolor": "white",
            "edgecolor": COLOR_ENTRY,
            "linewidth": 0.9,
            "alpha": 0.95,
        },
        zorder=6,
    )
    axes[1].annotate(
        "TIMING SHIFT: Sharpe ▼, win rate ▼, trade count flat —\n"
        "same number of trades, all of them worse.",
        xy=(0.015, 0.22),
        xycoords="axes fraction",
        ha="left",
        va="center",
        fontsize=8.6,
        color=COLOR_DRAWDOWN,
        bbox={
            "boxstyle": "round,pad=0.40",
            "facecolor": "white",
            "edgecolor": COLOR_DRAWDOWN,
            "linewidth": 0.9,
            "alpha": 0.95,
        },
        zorder=6,
    )

    axes[2].set_xlabel("leak dose (% of each sweep's full range)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_window_ratio_finding(
    half_life_rows: pd.DataFrame,
    ratio_rows: pd.DataFrame,
    out_path: str | Path,
    min_ratio: float = 5.0,
) -> Path:
    """Plot the window-to-half-life finding: the mechanism behind weak results.

    Two panels, because the finding needs both halves to be convincing:

    - **Left**: at *constant* signal-to-noise, Sharpe collapses as the half-life
      rises while the trade count barely moves. That rules out "fewer
      opportunities" and points at "worse ones".
    - **Right**: holding the data fixed and varying only the window recovers the
      performance. Same prices, same strategy, same costs — the estimator was
      the problem.

    The shaded band marks where ``window / half_life`` is below the usable
    threshold, which is where every real-world pair in this project landed under
    the default configuration.
    """
    out_path = Path(out_path)

    fig, (ax_hl, ax_ratio) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), layout="constrained"
    )

    # --- Left: Sharpe and win rate collapse with half-life ---
    ax_hl.plot(
        half_life_rows["half_life"], half_life_rows["sharpe"],
        color=COLOR_DRAWDOWN, linewidth=1.8, marker="o", markersize=5,
        label="Sharpe",
    )
    ax_hl.axhline(0.0, color=COLOR_NEUTRAL, linewidth=0.9)
    ax_hl.set_xlabel("spread half-life (bars)")
    ax_hl.set_ylabel("Sharpe ratio", color=COLOR_DRAWDOWN)
    ax_hl.tick_params(axis="y", labelcolor=COLOR_DRAWDOWN)
    ax_hl.set_title(
        "Slower reversion kills it — at constant signal-to-noise",
        loc="left", fontsize=11, fontweight="600",
    )
    _style_axes(ax_hl)

    ax_trades = ax_hl.twinx()
    ax_trades.plot(
        half_life_rows["half_life"], half_life_rows["win_rate"] * 100.0,
        color=COLOR_SPREAD, linewidth=1.3, linestyle="--", marker="s",
        markersize=4, label="win rate",
    )
    ax_trades.set_ylabel("win rate (%)", color=COLOR_SPREAD)
    ax_trades.tick_params(axis="y", labelcolor=COLOR_SPREAD)
    ax_trades.spines["top"].set_visible(False)

    n_min = int(half_life_rows["n_trades"].min())
    n_max = int(half_life_rows["n_trades"].max())
    ax_hl.annotate(
        f"trade count barely moves ({n_min}–{n_max})\nso it is not fewer chances — it is worse ones",
        xy=(0.97, 0.90), xycoords="axes fraction", ha="right", va="top",
        fontsize=8, color=COLOR_NEUTRAL,
    )

    # --- Right: the same data, fixed — only the window changes ---
    ax_ratio.axvspan(
        0, min_ratio, color=COLOR_DRAWDOWN, alpha=0.07, zorder=0,
    )
    ax_ratio.plot(
        ratio_rows["window_ratio"], ratio_rows["sharpe"],
        color=COLOR_ENTRY, linewidth=1.8, marker="o", markersize=5,
    )
    ax_ratio.axhline(0.0, color=COLOR_NEUTRAL, linewidth=0.9)
    ax_ratio.axvline(
        min_ratio, color=COLOR_DRAWDOWN, linewidth=1.0, linestyle=":",
    )
    ax_ratio.set_xlabel("z-score window ÷ half-life")
    ax_ratio.set_ylabel("Sharpe ratio")
    ax_ratio.set_title(
        "Identical data — only the window changed",
        loc="left", fontsize=11, fontweight="600", color=COLOR_ENTRY,
    )
    ax_ratio.annotate(
        "trailing mean contaminated\nby the deviation being measured",
        xy=(0.30, 0.30), xycoords="axes fraction", ha="center", va="center",
        fontsize=8, color=COLOR_DRAWDOWN,
    )
    ax_ratio.annotate(
        f"every real pair in this project\nlanded in here under the default window",
        xy=(0.97, 0.12), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=8, color=COLOR_NEUTRAL,
    )
    _style_axes(ax_ratio)

    fig.suptitle(
        "Why the strategy underperforms: the estimator, not the edge",
        fontsize=12.5, fontweight="600", x=0.01, ha="left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path
