"""CSV interchange with the OCaml engine.

The schema is documented in ``docs/SCHEMA.md`` and enforced on both sides: the
OCaml reader validates its header and rejects malformed rows, and the writers
here validate before writing. Validating on both sides is not redundant — it
means a corruption is caught at whichever boundary it crosses first, with a
message naming that boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import PRICE_COLUMNS


def write_prices(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a price file in the schema the OCaml engine expects.

    Validates before writing rather than after, so a bad frame never reaches
    disk and cannot be picked up by a later stage of the pipeline.

    Args:
        df: Must contain ``date``, ``price_a``, ``price_b``. Extra columns
            (such as the ground-truth ``true_spread``) are dropped, because the
            engine's parser rejects rows with the wrong field count — and, more
            importantly, the engine must never see ground truth.
        path: Destination.

    Returns:
        The path written.

    Raises:
        ValueError: On missing columns, non-positive or non-finite prices,
            or dates that are not strictly increasing.
    """
    path = Path(path)
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"price frame is missing columns: {missing}")

    out = df.loc[:, list(PRICE_COLUMNS)].copy()

    if len(out) == 0:
        raise ValueError("refusing to write an empty price file")

    for column in ("price_a", "price_b"):
        values = out[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{column} contains non-finite values")
        if np.any(values <= 0):
            n_bad = int(np.sum(values <= 0))
            raise ValueError(
                f"{column} contains {n_bad} non-positive value(s); "
                "the engine takes log prices"
            )

    dates = out["date"].astype(str)
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError(
            "dates must be strictly increasing with no duplicates "
            "(every trailing window in the engine assumes chronological order)"
        )
    out["date"] = dates

    path.parent.mkdir(parents=True, exist_ok=True)
    # Fixed precision so a regenerated file is byte-identical, which is what
    # makes the pipeline reproducible.
    out.to_csv(path, index=False, float_format="%.10f")
    return path


def read_prices(path: str | Path) -> pd.DataFrame:
    """Read a price file, applying the same validation as the engine."""
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    for column in ("price_a", "price_b"):
        if np.any(df[column].to_numpy(dtype=float) <= 0):
            raise ValueError(f"{path}: {column} contains non-positive values")
    return df


def read_bars(path: str | Path) -> pd.DataFrame:
    """Read the engine's per-bar audit trail.

    Optional statistics (``hedge_ratio``, ``spread``, ``zscore``) are written
    as empty fields during warm-up and parse to NaN, which is the intended
    representation: a warm-up bar is genuinely missing the statistic, and
    coercing it to 0 would make it indistinguishable from a bar where the
    z-score happened to be zero.
    """
    df = pd.read_csv(path)
    expected = {
        "index",
        "date",
        "price_a",
        "price_b",
        "hedge_ratio",
        "spread",
        "zscore",
        "position",
        "qty_a",
        "qty_b",
        "cash",
        "position_value",
        "nav",
        "costs_this_bar",
        "trade_event",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing bar columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    # An absent trade_event reads as NaN; normalise to the empty string so
    # downstream string operations do not need a null check.
    df["trade_event"] = df["trade_event"].fillna("")
    return df


def read_trades(path: str | Path) -> pd.DataFrame:
    """Read the engine's trade log. An empty log (no trades) is valid."""
    df = pd.read_csv(path)
    if len(df) == 0:
        return df
    for column in ("entry_date", "exit_date"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def read_metrics(path: str | Path) -> dict[str, float]:
    """Read the engine's metrics file into a dict.

    Values are parsed as floats where possible. A value that will not parse is
    kept as a string rather than being silently dropped, so a format change
    surfaces as a type error downstream instead of a missing key.
    """
    df = pd.read_csv(path)
    if not {"metric", "value"}.issubset(df.columns):
        raise ValueError(f"{path} must have columns 'metric' and 'value'")
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        raw = row["value"]
        try:
            out[str(row["metric"])] = float(raw)
        except (TypeError, ValueError):
            out[str(row["metric"])] = raw
    return out
