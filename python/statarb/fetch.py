"""Real market data via yfinance, with graceful degradation.

Network access is not guaranteed — CI runners are frequently sandboxed, and
yfinance's unofficial endpoint changes without notice. The pipeline therefore
treats real data as *optional*: if the fetch fails, the synthetic results are
still complete and the README says plainly that the real-data section was
skipped and why.

The failure mode this avoids is a pipeline that silently substitutes cached or
synthetic data when a fetch fails and then reports the result as if it came
from real prices.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd

#: Candidate pairs from sectors where a cointegrating relationship is
#: economically plausible: same industry, similar business model, exposed to
#: the same underlying commodity or rate.
#:
#: Naming these *ex ante* matters. Screening the whole S&P 500 for the best
#: cointegration p-value and then trading the winner is data snooping: with
#: ~125,000 pairs, thousands will pass a 5% test by chance alone. A short list
#: motivated by economics is the honest version of pair selection.
CANDIDATE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("KO", "PEP", "Beverages"),
    ("XOM", "CVX", "Integrated oil"),
    ("HD", "LOW", "Home improvement retail"),
    ("MA", "V", "Payment networks"),
    ("GS", "MS", "Investment banks"),
)

DEFAULT_START = "2015-01-01"
DEFAULT_END = "2024-12-31"


@dataclass(frozen=True)
class FetchResult:
    """Outcome of a data fetch.

    ``ok=False`` carries the reason, which is propagated verbatim into the
    README so a reader knows the real-data section is absent because the
    network was unavailable, not because the results were unflattering.
    """

    ok: bool
    frame: pd.DataFrame | None = None
    reason: str = ""


def fetch_pair(
    ticker_a: str,
    ticker_b: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> FetchResult:
    """Fetch adjusted daily closes for two tickers.

    Uses adjusted closes so that splits and dividends do not appear as price
    jumps. An unadjusted series would show a 2-for-1 split as a 50% overnight
    move, which the z-score would read as an enormous spread dislocation and
    trade into.

    Args:
        ticker_a: First ticker.
        ticker_b: Second ticker.
        start: ISO start date.
        end: ISO end date.

    Returns:
        A :class:`FetchResult`. On success, ``frame`` has columns ``date``,
        ``price_a``, ``price_b``, inner-joined on date so both legs are present
        on every bar.
    """
    try:
        import yfinance
    except ImportError:
        return FetchResult(ok=False, reason="yfinance is not installed")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yfinance.download(
                [ticker_a, ticker_b],
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
    except Exception as exc:  # noqa: BLE001 - any network/parse failure is the same outcome here
        return FetchResult(
            ok=False, reason=f"download failed for {ticker_a}/{ticker_b}: {exc}"
        )

    if raw is None or len(raw) == 0:
        return FetchResult(
            ok=False, reason=f"no data returned for {ticker_a}/{ticker_b}"
        )

    try:
        closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        series_a = closes[ticker_a]
        series_b = closes[ticker_b]
    except (KeyError, AttributeError) as exc:
        return FetchResult(
            ok=False, reason=f"unexpected response shape for {ticker_a}/{ticker_b}: {exc}"
        )

    # Inner join: a bar is usable only if both legs traded. Forward-filling a
    # missing leg would fabricate a price that never printed and, worse, would
    # hold the spread artificially constant across the gap.
    frame = pd.DataFrame({"price_a": series_a, "price_b": series_b}).dropna()
    if len(frame) < 200:
        return FetchResult(
            ok=False,
            reason=(
                f"only {len(frame)} overlapping bars for {ticker_a}/{ticker_b}; "
                "need at least 200"
            ),
        )
    if (frame <= 0).any().any():
        return FetchResult(
            ok=False, reason=f"non-positive prices in {ticker_a}/{ticker_b}"
        )

    frame = frame.reset_index()
    date_column = "Date" if "Date" in frame.columns else frame.columns[0]
    frame["date"] = pd.to_datetime(frame[date_column]).dt.strftime("%Y-%m-%d")
    return FetchResult(ok=True, frame=frame[["date", "price_a", "price_b"]])
