"""Chronological walk-forward splits with an embargo.

* **Expanding window.** Fold k trains on everything from the start of the data up to the start of
  its test period, then tests on the next ``test_fold_months``. Each fold re-trains, as a desk
  would periodically re-train a production model.
* **Embargo.** A training row is used only if its *target* interval had ended by the start of the
  test period (``target_time <= test_start``). Without this, the last few hours of training
  targets would overlap the first test intervals at long horizons.
* **Boundaries** are NEM month starts (00:00 AEST), expressed in UTC.
* Never random: shuffled splits would let the model train on the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nemspike.config import ValidationConfig
from nemspike.features.build import TARGET, TARGET_TIME
from nemspike.timeutils import NEM_TZ


@dataclass(frozen=True)
class Fold:
    number: int
    test_start: pd.Timestamp  # inclusive, UTC
    test_end: pd.Timestamp  # exclusive, UTC

    @property
    def label(self) -> str:
        start = self.test_start.tz_convert(NEM_TZ)
        end = (self.test_end - pd.Timedelta(minutes=5)).tz_convert(NEM_TZ)
        return f"{start:%Y-%m} to {end:%Y-%m}"


def _nem_month_start(ts: pd.Timestamp) -> pd.Timestamp:
    nem = ts.tz_convert(NEM_TZ)
    return pd.Timestamp(year=nem.year, month=nem.month, day=1, tz=NEM_TZ).tz_convert("UTC")


def walk_forward_folds(
    first_origin: pd.Timestamp, last_origin: pd.Timestamp, cfg: ValidationConfig
) -> list[Fold]:
    """Test folds covering everything after the initial training window up to ``last_origin``."""
    data_start = _nem_month_start(first_origin)
    test_start = (
        data_start.tz_convert(NEM_TZ) + pd.DateOffset(months=cfg.initial_train_months)
    ).tz_convert("UTC")
    folds: list[Fold] = []
    while test_start <= last_origin:
        test_end = (
            test_start.tz_convert(NEM_TZ) + pd.DateOffset(months=cfg.test_fold_months)
        ).tz_convert("UTC")
        folds.append(Fold(len(folds), test_start, min(test_end, last_origin + pd.Timedelta(1))))
        test_start = test_end
    if not folds:
        raise ValueError("not enough data for one test fold after the initial training window")
    return folds


def split(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and test rows for a fold. Rows with an unknown target are dropped from both."""
    has_target = df[TARGET].notna()
    train = df[has_target & (df[TARGET_TIME] <= fold.test_start)]
    test = df[has_target & (df.index >= fold.test_start) & (df.index < fold.test_end)]
    return train, test
