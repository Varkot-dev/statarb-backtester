"""Python layer for the statistical arbitrage backtester.

Responsibilities are split from the OCaml engine along a deliberate line:

- **OCaml** owns the backtest engine: the event loop, rolling statistics,
  signal generation, and position/PnL accounting. Everything where a
  correctness bug would silently produce a plausible-looking wrong number.
- **Python** owns data ingestion, cointegration research and pair selection,
  and plotting. Everything where the ecosystem (``statsmodels``,
  ``matplotlib``) is genuinely better than anything worth hand-rolling.

The two communicate over CSV with a documented schema; see
``statarb.io`` and ``docs/SCHEMA.md``.
"""

__all__ = [
    "DATA_SCHEMA_VERSION",
    "PRICE_COLUMNS",
]

#: Bumped when the CSV interchange schema changes in a way that would break
#: the OCaml reader. The engine validates its header on every load, so a
#: mismatch is a loud failure rather than a silent misparse.
DATA_SCHEMA_VERSION = "1.0"

#: The exact header the OCaml engine expects for a price file.
PRICE_COLUMNS = ("date", "price_a", "price_b")
