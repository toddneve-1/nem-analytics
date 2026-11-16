"""Typed experiment configuration loaded from YAML.

Using pydantic means a typo or wrong type in the YAML fails loudly at load time rather than
silently producing a different experiment.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class WeatherConfig(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    model: str = "era5"
    availability_lag_hours: int = Field(ge=0)


class DataConfig(BaseModel):
    region: str
    start: date
    end: date
    raw_dir: Path
    processed_dir: Path
    weather: WeatherConfig

    @model_validator(mode="after")
    def _check_dates(self) -> DataConfig:
        if self.end < self.start:
            raise ValueError("data.end must not be before data.start")
        return self


class ForecastConfig(BaseModel):
    horizons_minutes: list[int]
    dispatch_horizons_minutes: list[int]
    quantiles: list[float]
    target_transform: Literal["none", "asinh"]
    asinh_scale: float = Field(gt=0)

    @field_validator("horizons_minutes", "dispatch_horizons_minutes")
    @classmethod
    def _five_minute_multiples(cls, v: list[int]) -> list[int]:
        if any(h <= 0 or h % 5 for h in v):
            raise ValueError("horizons must be positive multiples of 5 minutes")
        return sorted(v)

    @field_validator("quantiles")
    @classmethod
    def _valid_quantiles(cls, v: list[float]) -> list[float]:
        if any(not 0 < q < 1 for q in v):
            raise ValueError("quantiles must lie strictly between 0 and 1")
        if 0.5 not in v:
            raise ValueError("quantiles must include 0.5 (the median forecast)")
        return sorted(v)


class FeaturesConfig(BaseModel):
    price_lags_intervals: list[int]
    rolling_windows_minutes: list[int]
    count_above_thresholds: list[float]
    weather_builders: list[str]

    @field_validator("price_lags_intervals")
    @classmethod
    def _non_negative_lags(cls, v: list[int]) -> list[int]:
        if any(k < 0 for k in v):
            raise ValueError("lags must be >= 0 (0 is the latest completed interval)")
        return sorted(set(v))

    @field_validator("rolling_windows_minutes")
    @classmethod
    def _window_multiples(cls, v: list[int]) -> list[int]:
        if any(w <= 5 or w % 5 for w in v):
            raise ValueError("rolling windows must be multiples of 5 minutes and > 5")
        return sorted(set(v))


class SpikeConfig(BaseModel):
    method: Literal["fixed", "percentile"]
    fixed_threshold: float
    percentile: float = Field(gt=0, lt=100)


class ValidationConfig(BaseModel):
    initial_train_months: int = Field(ge=1)
    test_fold_months: int = Field(ge=1)


class BatteryConfig(BaseModel):
    power_mw: float = Field(gt=0)
    energy_mwh: float = Field(gt=0)
    round_trip_efficiency: float = Field(gt=0, le=1)
    soc_min_frac: float = Field(ge=0, le=1)
    soc_max_frac: float = Field(ge=0, le=1)
    soc_initial_frac: float = Field(ge=0, le=1)
    rolling_resolve_minutes: int = Field(gt=0)
    lookahead_hours: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_soc(self) -> BatteryConfig:
        if not self.soc_min_frac <= self.soc_initial_frac <= self.soc_max_frac:
            raise ValueError("require soc_min_frac <= soc_initial_frac <= soc_max_frac")
        return self


class TrackingConfig(BaseModel):
    mlflow_tracking_uri: str
    experiment_name: str


class Config(BaseModel):
    seed: int
    data: DataConfig
    forecast: ForecastConfig
    features: FeaturesConfig
    spike: SpikeConfig
    validation: ValidationConfig
    battery: BatteryConfig
    tracking: TrackingConfig


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load and validate a YAML config file."""
    with Path(path).open() as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
