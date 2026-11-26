"""Baseline forecasters.

Point baselines (the classic naive forecasts):

* **persistence**: the latest known price, ``rrp_lag_0``;
* **same time yesterday**: price at the target's time of day one day earlier;
* **same time last week**: the same, one week earlier.

To compare fairly with quantile models, each point baseline is also made probabilistic: its
quantiles are the point forecast plus the empirical quantiles of its own training-fold errors,
estimated separately for each target hour of day (errors at 6 pm differ from errors at 3 am).
Its spike probability is the training-fold spike rate given whether the reference price was itself
a spike.

**Climatology** ignores recent history: quantiles and spike rates of training prices by target hour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nemspike.evaluation.labels import spike_labels
from nemspike.features.build import TARGET
from nemspike.models.base import Forecast, Forecaster, qcol

GROUP = "target_hour"


def _grouped_quantiles(
    values: pd.Series, groups: pd.Series, quantiles: list[float]
) -> pd.DataFrame:
    """Quantiles of ``values`` per group, plus a pooled fallback row under key -1."""
    table = values.groupby(groups).quantile(quantiles).unstack()
    table.loc[-1] = values.quantile(quantiles).to_numpy()
    table.columns = [qcol(q) for q in quantiles]
    return table


def _lookup(table: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    keys = groups.where(groups.isin(table.index), -1)
    out = table.loc[keys.to_numpy()].copy()
    out.index = groups.index
    return out


class PointBaseline(Forecaster):
    """A naive point forecast from one feature column, with empirical-error quantiles."""

    def __init__(self, name: str, column: str, quantiles: list[float]) -> None:
        self.name = name
        self.column = column
        self.quantiles = sorted(quantiles)
        self._residual_q: pd.DataFrame | None = None
        self._spike_rate: dict[bool, float] = {}
        self._threshold = float("nan")

    def fit(self, train: pd.DataFrame, spike_threshold: float) -> PointBaseline:
        rows = train[[self.column, TARGET, GROUP]].dropna()
        residual = rows[TARGET] - rows[self.column]
        self._residual_q = _grouped_quantiles(residual, rows[GROUP], self.quantiles)
        self._threshold = spike_threshold
        label = spike_labels(rows[TARGET], spike_threshold)
        ref_spike = rows[self.column] > spike_threshold
        overall = float(label.mean())
        self._spike_rate = {
            flag: float(label[ref_spike == flag].mean()) if (ref_spike == flag).any() else overall
            for flag in (False, True)
        }
        return self

    def predict(self, test: pd.DataFrame) -> Forecast:
        if self._residual_q is None:
            raise RuntimeError("fit() must be called before predict()")
        point = test[self.column]
        offsets = _lookup(self._residual_q, test[GROUP])
        quantiles = offsets.add(point, axis=0)
        ref_spike = point > self._threshold
        prob = pd.Series(
            np.where(ref_spike, self._spike_rate[True], self._spike_rate[False]), index=test.index
        )
        return Forecast(point=point, quantiles=quantiles, spike_prob=prob)


class Climatology(Forecaster):
    """Quantiles and spike rate of training-fold prices by target hour of day."""

    name = "climatology"

    def __init__(self, quantiles: list[float]) -> None:
        self.quantiles = sorted(set(quantiles) | {0.5})
        self._table: pd.DataFrame | None = None
        self._spike_rate: pd.Series | None = None

    def fit(self, train: pd.DataFrame, spike_threshold: float) -> Climatology:
        rows = train[[TARGET, GROUP]].dropna()
        self._table = _grouped_quantiles(rows[TARGET], rows[GROUP], self.quantiles)
        label = spike_labels(rows[TARGET], spike_threshold)
        rate = label.groupby(rows[GROUP]).mean()
        rate.loc[-1] = label.mean()
        self._spike_rate = rate
        return self

    def predict(self, test: pd.DataFrame) -> Forecast:
        if self._table is None or self._spike_rate is None:
            raise RuntimeError("fit() must be called before predict()")
        q = _lookup(self._table, test[GROUP])
        keys = test[GROUP].where(test[GROUP].isin(self._spike_rate.index), -1)
        prob = pd.Series(self._spike_rate.loc[keys.to_numpy()].to_numpy(), index=test.index)
        return Forecast(point=q[qcol(0.5)], quantiles=q, spike_prob=prob)


def default_baselines(quantiles: list[float]) -> list[Forecaster]:
    return [
        PointBaseline("persistence", "rrp_lag_0", quantiles),
        PointBaseline("same_time_yesterday", "rrp_same_time_yesterday", quantiles),
        PointBaseline("same_time_last_week", "rrp_same_time_last_week", quantiles),
        Climatology(quantiles),
    ]
