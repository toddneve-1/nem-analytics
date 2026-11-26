"""Common interface for all forecasters (baselines and machine-learning models)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


def qcol(q: float) -> str:
    """Column name for a quantile level, for example 0.9 -> 'q0.9'."""
    return f"q{q:g}"


@dataclass
class Forecast:
    """Predictions for a set of origins (all indexed like the test rows).

    ``point`` is the forecast scored by MAE/RMSE. ``quantiles`` has one column per level
    (``q0.1`` etc.) and is optional for point-only models. ``spike_prob`` is P(price > threshold).
    """

    point: pd.Series
    quantiles: pd.DataFrame | None = None
    spike_prob: pd.Series | None = None


class Forecaster(ABC):
    """Fit on a training fold, predict a test fold. Must use feature columns only."""

    name: str

    @abstractmethod
    def fit(self, train: pd.DataFrame, spike_threshold: float) -> Forecaster: ...

    @abstractmethod
    def predict(self, test: pd.DataFrame) -> Forecast: ...
