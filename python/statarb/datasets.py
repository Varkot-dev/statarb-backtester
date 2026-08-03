"""Dataset definitions for the reproducible pipeline.

Every dataset the README reports on is declared here with its seed and
parameters, so ``make backtest`` regenerates byte-identical inputs and the
reported numbers are reproducible by a third party.

The three synthetic datasets are chosen to span the range of outcomes a pairs
strategy can produce, including the ones that make it look bad:

- ``coint_medium``: a genuinely cointegrated pair with a tradeable half-life.
  The headline result.
- ``coint_slow``: cointegrated but slow-reverting. Positions hit the holding
  cap more often; shows how the strategy degrades when the signal is real but
  the timescale is wrong.
- ``independent``: two independent random walks. **The negative control.**
  There is no relationship to trade, so any apparent edge here is the
  backtester lying. Reporting this is the point.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .synth import CointegratedSpec, generate_cointegrated, generate_independent

#: Bars generated for each synthetic dataset. Roughly ten business years, long
#: enough that the metrics are not dominated by a handful of trades and that a
#: 60-bar warm-up is a small fraction of the sample.
DEFAULT_N_BARS = 2520

#: Master seed. Every dataset derives its seed from this, so the entire corpus
#: is reproducible from one number.
MASTER_SEED = 20250917


@dataclass(frozen=True)
class Dataset:
    """A named, reproducible dataset.

    Attributes:
        key: Short identifier, used for filenames and report sections.
        title: Human-readable description for the README.
        kind: ``"cointegrated"`` or ``"independent"``.
        seed: PRNG seed.
        spec: Ground-truth parameters, for cointegrated datasets.
        n_bars: Length, for independent datasets.
        is_negative_control: Whether a good result here would indicate a bug
            rather than a discovery.
    """

    key: str
    title: str
    kind: str
    seed: int
    spec: CointegratedSpec | None = None
    n_bars: int = DEFAULT_N_BARS
    is_negative_control: bool = False

    def generate(self) -> pd.DataFrame:
        """Materialise the dataset."""
        if self.kind == "cointegrated":
            if self.spec is None:
                raise ValueError(f"dataset {self.key} is cointegrated but has no spec")
            return generate_cointegrated(self.spec, seed=self.seed)
        if self.kind == "independent":
            return generate_independent(self.n_bars, seed=self.seed)
        raise ValueError(f"unknown dataset kind: {self.kind}")


#: The datasets the pipeline runs. Parameters were fixed before any backtest
#: was run; they are not tuned to produce an attractive equity curve.
SYNTHETIC_DATASETS: tuple[Dataset, ...] = (
    Dataset(
        key="coint_medium",
        title="Cointegrated pair, 15-bar half-life (primary)",
        kind="cointegrated",
        seed=MASTER_SEED,
        spec=CointegratedSpec(
            n_bars=DEFAULT_N_BARS,
            beta=1.20,
            half_life=15.0,
            sigma_spread=0.030,
            sigma_common=0.011,
        ),
    ),
    Dataset(
        key="coint_slow",
        title="Cointegrated pair, 45-bar half-life (slow reversion)",
        kind="cointegrated",
        seed=MASTER_SEED + 1,
        spec=CointegratedSpec(
            n_bars=DEFAULT_N_BARS,
            beta=0.85,
            half_life=45.0,
            sigma_spread=0.035,
            sigma_common=0.011,
        ),
    ),
    Dataset(
        key="independent",
        title="Two independent random walks (NEGATIVE CONTROL)",
        kind="independent",
        seed=MASTER_SEED + 2,
        n_bars=DEFAULT_N_BARS,
        is_negative_control=True,
    ),
)


def dataset_by_key(key: str) -> Dataset:
    """Look up a dataset by key.

    Raises:
        KeyError: With the list of valid keys, since a typo in the Makefile
            should say what the options were.
    """
    for dataset in SYNTHETIC_DATASETS:
        if dataset.key == key:
            return dataset
    valid = ", ".join(d.key for d in SYNTHETIC_DATASETS)
    raise KeyError(f"unknown dataset '{key}'; valid keys are: {valid}")
