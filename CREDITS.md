# Credits and prior art

This repository is MIT licensed (see [`LICENSE`](LICENSE)) — chosen deliberately,
given that the section below declines to use another project specifically
because it carries no licence. It would be incoherent to make that argument and
then leave this repository in the same state.

Everything in this repository is original code. Nothing was copied. This file
records the external work that *informed* it, what was taken (ideas, not code),
and — where relevant — what was deliberately rejected and why.

## Ideas adopted

### `mnemox-ai/deflated-sharpe` (Apache-2.0)

**What was taken: the skewness/kurtosis correction to the Sharpe standard
error.** My first implementation used the Gaussian form,
`SE = sqrt(1/T)`. Reading this repository prompted switching to the
Mertens (2002) / Lo (2002) form used by Bailey & López de Prado:

```text
SE = sqrt((1 - γ₃·SR + (γ₄-1)/4·SR²) / (T-1))
```

This matters here specifically. Pairs-trading returns are far from normal — the
NAV returns in this project have **kurtosis ≈ 14** against a normal value of 3,
because the strategy produces many small mean-reversion wins and occasional
large stop-loss losses. The Gaussian standard error understates uncertainty on
exactly that distribution, in the direction that flatters the result.

Implemented independently in
[`statarb.diagnostics.sharpe_standard_error`](python/statarb/diagnostics.py).

**What was rejected: their `deflated_sharpe_ratio` function.** It documents
`observed_sr` as annualized, but compares it against an un-annualized threshold
(`e_z_max * sqrt(1/(T-1))`) and divides by an un-annualized standard error. The
units do not match, and the statistic is inflated by `sqrt(252) ≈ 15.9`.

Feeding it a realistic annualized Sharpe of 0.654 on 2,515 observations returns
**DSR = +28.0**, which would be the most statistically significant result in the
history of finance. The correct value is **−0.007** (p = 0.50) — a coin flip,
which is exactly right for a result found by searching 30 configurations.

[`deflated_sharpe_threshold`](python/statarb/diagnostics.py) does all arithmetic
in per-bar units and converts once at the end, and
`test_units_are_handled_consistently` pins the bound so the mistake cannot
reappear here.

This is not a criticism of the project — the idea was worth having and the
correction is genuinely better than what I had. It is recorded because "I
adopted the concept but not the implementation, for this specific reason" is the
honest description of what happened.

### `eslazarev/purged-cross-validation` (MIT)

Surveyed for its treatment of purging and embargo in time-series
cross-validation. Not used: this project's walk-forward split is a single
in-sample/out-of-sample boundary rather than a k-fold scheme, so purging around
fold edges does not apply. Recorded because it is the right reference if this
were extended to combinatorially purged CV.

## Deliberately not used

### `Aliipou/backtest-audit`

The closest match to this project's needs (Deflated Sharpe + Probability of
Backtest Overfitting + Monte Carlo permutation tests), but it carries **no
licence file**.

On GitHub, absent a licence, default copyright applies: the work is *all rights
reserved*. Public visibility grants the right to view and to fork within GitHub,
not to copy the code into another project. So it could not be used here
regardless of how well it fit.

### `quantskills/skill-backtest-overfit` (GPL-3.0)

Implements Deflated Sharpe, PBO via CSCV, and the Harvey–Liu haircut. Not used:
GPL-3.0 is copyleft and would require this repository to adopt the same licence.
That is a legitimate choice for a project to make, but not one to make by
accident while borrowing a helper function.

## Academic references

The methods here are implementations of published work:

- **Gatev, Goetzmann & Rouwenhorst (2006)**, *Pairs Trading: Performance of a
  Relative-Value Arbitrage Rule* — the 2σ entry convention, and the documented
  post-publication decay this project's real-data results are consistent with.
- **Granger & Newbold (1974)**, *Spurious Regressions in Econometrics* — why
  level correlation between I(1) series is misleading. Demonstrated on real data
  in the README (KO/PEP: 0.978 correlation, not cointegrated).
- **Engle & Granger (1987)**, *Co-integration and Error Correction* — the
  two-step test, used as the primary selection criterion.
- **Johansen (1991)** — the symmetric maximum-likelihood alternative, required
  to agree with Engle-Granger before a pair is accepted.
- **Lo (2002)**, *The Statistics of Sharpe Ratios* — standard errors, and why
  serial correlation breaks √T scaling.
- **Mertens (2002)** — the non-normality correction to the Sharpe standard
  error.
- **Politis & Romano (1994)**, *The Stationary Bootstrap* — block resampling for
  dependent data, used for the Sharpe confidence intervals.
- **Bailey & López de Prado (2014)**, *The Deflated Sharpe Ratio* — correcting
  for selection bias under multiple testing. This is the correction that reduced
  this project's own apparent real-data discovery to noise.

## Libraries

- `statsmodels` — ADF, Engle-Granger `coint`, and `coint_johansen`. The
  cointegration tests are not reimplemented; there is no reason to hand-roll
  critical-value tables.
- `numpy` / `pandas` — numerics and data handling.
- `matplotlib` — charts.
- OCaml: `dune`, `alcotest`, `qcheck`. The engine itself has no runtime
  dependencies beyond the standard library.

## Prior-art search for the window/half-life finding

The substantive claim in this project — that the ratio of the z-score window to
the spread's half-life governs whether a pairs signal works, with a usable
threshold around 5× — was searched for before being written up:

```bash
gh search code "half_life" "zscore window"
gh search repos "ornstein uhlenbeck half life window pairs"
```

Neither returned anything relating the two quantities. One result
(`Fisjo/imc-prosperity-4`) computes both a pair half-life and a z-score window
in the same file without connecting them.

**This is weak evidence, not proof.** A GitHub code search does not cover the
academic literature, and the relationship is plausibly folklore among
practitioners even if it is not written down in public code. The README states
the finding as a measured result on this project's own data, which it is,
without claiming priority.
