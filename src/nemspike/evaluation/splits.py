"""Chronological walk-forward splits with an embargo.

* **Expanding window.** Fold k trains on everything from the start of the data up to the start of
  its test period, then tests on the next ``test_fold_months``. Each fold re-trains, as a desk
  would periodically re-train a production model.
* **Embargo.** A training row is used only if its *target* interval had ended by the start of the
  test period (``target_time <= test_start``). Without this, the last few hours of training
  targets would overlap the first test intervals at long horizons.
* **Boundaries** are NEM month starts (00:00 AEST), expressed in UTC.
* Never random: shuffled splits would let the model train on the future.
* **Final holdout.** If ``holdout_start`` is set, development folds stop there. The holdout is a
  single extra fold, trained on everything before it, and is only produced on request
  (``stage="holdout"``), so model choices cannot be tuned on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from nemspike.config import ValidationConfig
from nemspike.features.build import TARGET, TARGET_TIME
from nemspike.timeutils import NEM_TZ, nem_day_start_utc

Stage = Literal["dev", "holdout"]


@dataclass(frozen=True)
class Fold:
    number: int
    test_start: pd.Timestamp  # inclusive, UTC
    test_end: pd.Timestamp  # exclusive, UTC
    is_holdout: bool = False

    @property
    def label(self) -> str:
        start = self.test_start.tz_convert(NEM_TZ)
        end = (self.test_end - pd.Timedelta(minutes=5)).tz_convert(NEM_TZ)
        prefix = "holdout " if self.is_holdout else ""
        return f"{prefix}{start:%Y-%m} to {end:%Y-%m}"


def _nem_month_start(ts: pd.Timestamp) -> pd.Timestamp:
    nem = ts.tz_convert(NEM_TZ)
    return pd.Timestamp(year=nem.year, month=nem.month, day=1, tz=NEM_TZ).tz_convert("UTC")


def walk_forward_folds(
    first_origin: pd.Timestamp,
    last_origin: pd.Timestamp,
    cfg: ValidationConfig,
    stage: Stage = "dev",
) -> list[Fold]:
    """Test folds after the initial training window.

    ``stage="dev"``: quarterly folds up to the holdout start (or ``last_origin`` if no holdout).
    ``stage="holdout"``: one fold from the holdout start to ``last_origin``.
    """
    end = last_origin + pd.Timedelta(1)
    holdout_start = nem_day_start_utc(cfg.holdout_start) if cfg.holdout_start else None
    if stage == "holdout":
        if holdout_start is None:
            raise ValueError("no holdout configured (validation.holdout_start)")
        if holdout_start > last_origin:
            raise ValueError("holdout starts after the last origin")
        return [Fold(0, holdout_start, end, is_holdout=True)]

    dev_end = min(end, holdout_start) if holdout_start is not None else end
    data_start = _nem_month_start(first_origin)
    test_start = (
        data_start.tz_convert(NEM_TZ) + pd.DateOffset(months=cfg.initial_train_months)
    ).tz_convert("UTC")
    folds: list[Fold] = []
    # A fold must contain an origin with a known target. The last origin never has one, so a
    # fold starting exactly there (data ending on a fold boundary, no holdout) would be empty.
    while test_start < min(dev_end, last_origin):
        test_end = (
            test_start.tz_convert(NEM_TZ) + pd.DateOffset(months=cfg.test_fold_months)
        ).tz_convert("UTC")
        folds.append(Fold(len(folds), test_start, min(test_end, dev_end)))
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
