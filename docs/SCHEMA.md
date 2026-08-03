# CSV interchange schema

The OCaml engine and the Python research layer communicate over CSV files with
the schemas below. Both sides validate: the OCaml reader checks its header and
rejects any malformed row with a line number, and the Python writer validates
before writing so a bad frame never reaches disk.

Validating on both sides is not redundant. It means a corruption is caught at
whichever boundary it crosses first, with an error naming that boundary,
instead of surfacing three stages later as a puzzling number.

**Schema version: 1.0** (`statarb.DATA_SCHEMA_VERSION`)

---

## Input: price series

**Written by** `statarb.io.write_prices` · **read by** `Csv_io.read_prices`

```csv
date,price_a,price_b
2015-01-01,100.0000000000,50.0000000000
2015-01-02,100.7183294837,50.1128374650
```

| Column | Type | Constraints |
| --- | --- | --- |
| `date` | ISO-8601 `YYYY-MM-DD` | Non-empty; **strictly increasing**, no duplicates |
| `price_a` | float | **Strictly positive** and finite |
| `price_b` | float | **Strictly positive** and finite |

### Why these constraints are enforced rather than assumed

**Strictly increasing dates.** Every trailing window in the engine assumes the
array is chronological. Checking it once at the boundary means no rolling
statistic has to re-verify it, and a shuffled or duplicated file fails
immediately rather than producing a subtly wrong z-score.

**Strictly positive prices.** The engine takes logs when building the spread. A
zero or negative price would produce `-inf` or `nan` that propagates silently
through the hedge-ratio regression into the reported metrics. Rejecting at load
turns a silent corruption into a loud one.

**Exactly three columns.** The parser rejects rows with a different field
count. This is what stops ground-truth columns (`true_spread`, `common_trend`)
from reaching the engine — which would be the ultimate lookahead. `write_prices`
drops them for the same reason.

Dates are treated as **opaque labels** by the engine. It never does date
arithmetic; annualization uses the `bars_per_year` constant instead. This is why
the fixtures can use synthetic sequential dates without affecting any result.

---

## Output: per-bar audit trail (`bars.csv`)

**Written by** `Csv_io.write_bars` · **read by** `statarb.io.read_bars`

One row per input bar. This file is the point of the whole design: **every
number the README reports can be re-derived from it**, so a third party never
has to trust the engine's own summary.

| Column | Type | Notes |
| --- | --- | --- |
| `index` | int | 0-based bar index |
| `date` | string | Copied from the input |
| `price_a`, `price_b` | float | Copied from the input |
| `hedge_ratio` | float or **empty** | Rolling OLS β; empty during warm-up |
| `spread` | float or **empty** | OLS residual; empty during warm-up |
| `zscore` | float or **empty** | Rolling z-score; empty during warm-up or on a degenerate window |
| `position` | enum | `flat` \| `long_spread` \| `short_spread` |
| `qty_a`, `qty_b` | float | **Signed** share counts (negative = short) |
| `cash` | float | Cash balance |
| `position_value` | float | Signed mark-to-market of the open legs |
| `nav` | float | `cash + position_value` |
| `costs_this_bar` | float | Commission + slippage paid this bar |
| `interest_this_bar` | float | Risk-free interest credited on cash |
| `trade_event` | string | `""` \| `entry:<dir>` \| `exit:<reason>` \| `flip:<dir>` |

### Empty versus zero

Optional statistics are written as **empty fields**, never as `0` or `nan`.
Pandas reads an empty field as `NaN`, which is the correct representation of
"this statistic did not exist yet". Writing `0` would make a warm-up bar
indistinguishable from a bar where the z-score genuinely was zero.

### The accounting identities

These hold at every bar and are asserted both inside the engine (which aborts
on failure) and from the outside in the test suites:

```text
nav[t] = cash[t] + position_value[t]

nav[t] = nav[t-1]
       + qty_a[t-1] * (price_a[t] - price_a[t-1])     # mark-to-market, leg A
       + qty_b[t-1] * (price_b[t] - price_b[t-1])     # mark-to-market, leg B
       + interest_this_bar[t]
       - costs_this_bar[t]
```

Note the identity uses `qty[t-1]` — the position held *into* the bar, which is
the one exposed to the bar's price move. Using `qty[t]` is the subtle version
of the bug these checks exist to catch.

### Verifying next-bar execution from this file

For any row where `trade_event` starts with `entry`, the **previous** row's
`zscore` must already breach the entry threshold. If a fill ever coincided with
the bar whose z-score triggered it, that would be lookahead — and this file is
enough to check it without reading any source code.

---

## Output: trade log (`trades.csv`)

**Written by** `Csv_io.write_trades` · **read by** `statarb.io.read_trades`

One row per **closed** round-trip trade. An empty file (header only) is valid:
a strategy that never traded is a legitimate outcome, not an error.

| Column | Type | Notes |
| --- | --- | --- |
| `direction` | enum | `long_spread` \| `short_spread` |
| `entry_index`, `exit_index` | int | Bar indices of the fills |
| `entry_date`, `exit_date` | string | Corresponding dates |
| `entry_z`, `exit_z` | float | z-scores that triggered each side |
| `pnl_gross` | float | Before costs |
| `costs` | float | Entry + exit, both legs |
| `pnl_net` | float | `pnl_gross - costs`, exactly |
| `exit_reason` | enum | `reversion` \| `stop_loss` \| `max_holding` \| `end_of_data` |
| `holding_bars` | int | `exit_index - entry_index` |

`exit_reason` is worth reading. A strategy exiting mostly on `stop_loss` is not
the mean-reversion strategy it claims to be, whatever its Sharpe says.

---

## Output: metrics (`metrics.csv`)

**Written by** `Csv_io.write_metrics` · **read by** `statarb.io.read_metrics`

```csv
metric,value
sharpe_ratio,0.7394821053
max_drawdown,-0.0126394821
n_trades,54
```

Two columns, one metric per row. Formatted at 10 decimal places so a
regenerated file is byte-identical.

Key metrics — the full list is in `Metrics.to_csv_rows`:

| Metric | Notes |
| --- | --- |
| `sharpe_ratio` | Annualized by `sqrt(bars_per_year)`; see the README on why this overstates |
| `sharpe_per_bar` | Un-annualized, for readers who want to apply their own factor |
| `max_drawdown` | **Negative** fraction (`-0.05` = a 5% decline) |
| `total_interest` | Interest credited on idle cash |
| `total_costs` | All commission and slippage |
| `n_reversion`, `n_stop_loss`, `n_max_holding`, `n_end_of_data` | Must sum to `n_trades` |

---

## Output: parameter sweep (`sweep.csv`)

One row per parameter combination, from `statarb sweep`.

```csv
label,window,entry,exit,sharpe_ratio,annualized_return,max_drawdown,n_trades,win_rate,total_costs,final_nav
```

---

## Output: pipeline summary (`summary.json`)

Written by `scripts/run_pipeline.py`. Aggregates every dataset's metrics,
cointegration diagnostics, significance tests, and chart paths — the structure
the README's results section is built from.

```json
{
  "synthetic": [
    {
      "key": "coint_medium",
      "title": "...",
      "is_negative_control": false,
      "metrics": { "sharpe_ratio": 0.739, "...": "..." },
      "cointegration": { "eg_pvalue": 0.0, "true_beta": 1.2, "...": "..." },
      "significance": { "t_statistic": 2.788, "p_value": 0.007, "...": "..." },
      "charts": { "equity": "reports/coint_medium/equity_curve.png" }
    }
  ],
  "cash_interest_comparison": { "with_interest": {}, "without_interest": {} },
  "sweep_csv": "reports/sweep.csv",
  "real_data": { "available": true, "attempts": [], "results": [] }
}
```

---

## Changing the schema

Both sides validate headers, so a change made on one side alone fails loudly
rather than silently misparsing. The steps:

1. Update the OCaml writer/reader in `engine/lib/csv_io.ml`.
2. Update the Python reader in `python/statarb/io.py`.
3. Update the fixtures in `engine/test/test_csv_io.ml` and
   `python/tests/test_io.py`.
4. Bump `DATA_SCHEMA_VERSION` in `python/statarb/__init__.py`.
5. Update this document.

The cross-language tests in `python/tests/test_integration.py` are what catch a
drift between steps 1 and 2.
