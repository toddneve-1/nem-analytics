"""Assemble the modelling table for one forecast horizon.

Each row is a forecast origin ``T``. Columns are the features available at ``T`` plus:

* ``target``: the RRP of the interval ending at ``T + horizon`` (raw $/MWh, not transformed);
* ``target_time``: that interval's end time (UTC).

Target transforms and spike labels are applied later, inside each training fold, so that any
fitted quantity (for example a percentile threshold) only ever sees training data.
"""

from __future__ import annotations

import pandas as pd

from nemspike.config import Config
from nemspike.features.calendar import calendar_features
from nemspike.features.price import (
    INTERVALS_PER_WEEK,
    history_features,
    seasonal_features,
    to_regular_grid,
)
from nemspike.features.weather import build_weather_features

TARGET = "target"
TARGET_TIME = "target_time"
NON_FEATURE_COLUMNS = (TARGET, TARGET_TIME)


def make_target(grid: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    """RRP of the interval ending ``horizon_minutes`` after each origin."""
    if horizon_minutes <= 0 or horizon_minutes % 5:
        raise ValueError("horizon must be a positive multiple of 5 minutes")
    return grid["rrp"].shift(-(horizon_minutes // 5)).rename(TARGET)


def build_features(
    prices: pd.DataFrame,
    weather: pd.DataFrame | None,
    horizon_minutes: int,
    cfg: Config,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """Feature table for one horizon, indexed by origin.

    ``prices`` is the AEMO table (``interval_end``, ``rrp``, ``demand_mw``). ``weather`` is the
    hourly table from :mod:`nemspike.data.weather`, or ``None`` to build without weather.
    Rows whose target lies beyond the data have ``target`` = NaN (useful for live forecasting).
    With ``drop_warmup``, the first week is dropped because weekly lags are undefined there.
    """
    grid = to_regular_grid(prices)
    origins = grid.index

    parts = [
        history_features(grid, cfg.features),
        seasonal_features(grid, horizon_minutes),
        calendar_features(
            origins, horizon_minutes, cfg.data.weather.latitude, cfg.data.weather.longitude
        ),
    ]
    if weather is not None and cfg.features.weather_builders:
        lag = pd.Timedelta(hours=cfg.data.weather.availability_lag_hours)
        parts.append(build_weather_features(weather, origins, cfg.features.weather_builders, lag))

    df = pd.concat(parts, axis=1)
    df[TARGET] = make_target(grid, horizon_minutes)
    df[TARGET_TIME] = origins + pd.Timedelta(minutes=horizon_minutes)
    if drop_warmup:
        df = df.iloc[INTERVALS_PER_WEEK:]
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns that are model inputs (everything except target bookkeeping)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
