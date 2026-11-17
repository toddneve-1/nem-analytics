"""Typed experiment configuration loaded from YAML.

Using pydantic means a typo or wrong type in the YAML fails loudly at load time rather than
silently producing a different experiment.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    """Base for every config section: an unknown key is an error, not silently ignored.

    Without this, a misspelt optional key (for example ``degradation_cost_per_mhw``) would fall
    back to its default and quietly change the experiment.
    """

    model_config = ConfigDict(extra="forbid")


class WeatherConfig(_Strict):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    model: str = "era5"
    availability_lag_hours: int = Field(ge=0)


class DataConfig(_Strict):
    region: str
    start: date
    end: date
    raw_dir: Path
    weather: WeatherConfig

    @model_validator(mode="after")
    def _check_dates(self) -> DataConfig:
        if self.end < self.start:
            raise ValueError("data.end must not be before data.start")
        return self


class ForecastConfig(_Strict):
    horizons_minutes: list[int]
    # Horizons whose forecasts form the price path for rolling-horizon dispatch. Must be an even
    # grid (step, 2 x step, ...): each forecast prices one block of the planning window.
    dispatch_horizons_minutes: list[int]
    quantiles: list[float]
    # Target transform of the default LightGBM model (``lgbm_<transform>``), which is the model
    # used by ``battery-forecast``, ``train-final`` and ``snapshot-predictions`` unless overridden.
    target_transform: Literal["none", "asinh"]
    asinh_scale: float = Field(gt=0)

    @field_validator("horizons_minutes", "dispatch_horizons_minutes")
    @classmethod
    def _five_minute_multiples(cls, v: list[int]) -> list[int]:
        if any(h <= 0 or h % 5 for h in v):
            raise ValueError("horizons must be positive multiples of 5 minutes")
        return sorted(v)

    @field_validator("dispatch_horizons_minutes")
    @classmethod
    def _even_grid(cls, v: list[int]) -> list[int]:
        if not v or v != [v[0] * (i + 1) for i in range(len(v))]:
            raise ValueError("dispatch horizons must form an even grid: step, 2 x step, ...")
        return v

    @property
    def all_horizons_minutes(self) -> list[int]:
        """Every horizon the pipeline needs: the headline horizons plus the dispatch grid."""
        return sorted(set(self.horizons_minutes) | set(self.dispatch_horizons_minutes))

    @property
    def default_model(self) -> str:
        return f"lgbm_{self.target_transform}"

    @field_validator("quantiles")
    @classmethod
    def _valid_quantiles(cls, v: list[float]) -> list[float]:
        if any(not 0 < q < 1 for q in v):
            raise ValueError("quantiles must lie strictly between 0 and 1")
        if 0.5 not in v:
            raise ValueError("quantiles must include 0.5 (the median forecast)")
        return sorted(v)


class FeaturesConfig(_Strict):
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


class SpikeConfig(_Strict):
    method: Literal["fixed", "percentile"]
    fixed_threshold: float
    percentile: float = Field(gt=0, lt=100)
    # Probability above which a spike is "predicted", for precision and recall. PR-AUC does not
    # depend on it.
    decision_probability: float = Field(default=0.5, gt=0, lt=1)


class ValidationConfig(_Strict):
    initial_train_months: int = Field(ge=1)
    test_fold_months: int = Field(ge=1)
    # Start (NEM date) of the final holdout. Development folds stop here; the holdout is scored
    # once, at the end, by a model trained on everything before it. None disables the holdout.
    holdout_start: date | None = None


class ModelConfig(_Strict):
    """LightGBM settings shared by the quantile models and the spike classifier."""

    train_stride_minutes: int = Field(ge=5)
    validation_days: int = Field(ge=7)
    learning_rate: float = Field(gt=0, le=1)
    num_leaves: int = Field(ge=2)
    min_data_in_leaf: int = Field(ge=1)
    feature_fraction: float = Field(gt=0, le=1)
    bagging_fraction: float = Field(gt=0, le=1)
    bagging_freq: int = Field(ge=0)
    lambda_l2: float = Field(ge=0)
    max_rounds: int = Field(ge=1)
    early_stopping_rounds: int = Field(ge=1)
    # 0 = LightGBM default (all cores). Set lower when other heavy jobs share the machine:
    # LightGBM threads spin while waiting, so two oversubscribed jobs slow each other badly.
    num_threads: int = Field(default=0, ge=0)

    @field_validator("train_stride_minutes")
    @classmethod
    def _stride_multiple(cls, v: int) -> int:
        if v % 5:
            raise ValueError("train_stride_minutes must be a multiple of 5")
        return v


class BatteryConfig(_Strict):
    power_mw: float = Field(gt=0)
    energy_mwh: float = Field(gt=0)
    round_trip_efficiency: float = Field(gt=0, le=1)
    soc_min_frac: float = Field(ge=0, le=1)
    soc_max_frac: float = Field(ge=0, le=1)
    soc_initial_frac: float = Field(ge=0, le=1)
    rolling_resolve_minutes: int = Field(gt=0)
    lookahead_hours: float = Field(gt=0)
    # Wear cost in $ per MWh discharged; the optimiser only trades if the spread covers it.
    degradation_cost_per_mwh: float = Field(default=0.0, ge=0)
    # "continuous": state of charge carried across midnights (50 % only at the start and end of
    # each evaluation period). "daily": every day starts and ends at the initial level.
    mode: Literal["continuous", "daily"] = "continuous"

    @model_validator(mode="after")
    def _check_soc(self) -> BatteryConfig:
        if not self.soc_min_frac <= self.soc_initial_frac <= self.soc_max_frac:
            raise ValueError("require soc_min_frac <= soc_initial_frac <= soc_max_frac")
        return self

    @model_validator(mode="after")
    def _check_rolling(self) -> BatteryConfig:
        if self.rolling_resolve_minutes % 5:
            raise ValueError("rolling_resolve_minutes must be a multiple of 5")
        if abs(self.lookahead_hours * 12 - round(self.lookahead_hours * 12)) > 1e-9:
            raise ValueError("lookahead_hours must be a whole number of 5-minute intervals")
        if self.rolling_resolve_minutes > self.lookahead_hours * 60:
            raise ValueError("rolling_resolve_minutes must not exceed the look-ahead window")
        return self

    @property
    def resolve_intervals(self) -> int:
        """5-minute intervals executed between re-plans."""
        return self.rolling_resolve_minutes // 5

    @property
    def lookahead_intervals(self) -> int:
        """5-minute intervals in each planning window."""
        return round(self.lookahead_hours * 12)


class TrackingConfig(_Strict):
    mlflow_tracking_uri: str
    experiment_name: str


class Config(_Strict):
    seed: int
    data: DataConfig
    forecast: ForecastConfig
    features: FeaturesConfig
    spike: SpikeConfig
    validation: ValidationConfig
    model: ModelConfig
    battery: BatteryConfig
    tracking: TrackingConfig

    @model_validator(mode="after")
    def _check_dispatch_window(self) -> Config:
        longest = max(self.forecast.dispatch_horizons_minutes)
        if self.battery.lookahead_hours * 60 > longest:
            raise ValueError(
                "battery.lookahead_hours extends beyond the longest dispatch horizon "
                f"({longest} minutes): there would be no forecast for the end of each plan"
            )
        return self


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load and validate a YAML config file."""
    with Path(path).open() as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
