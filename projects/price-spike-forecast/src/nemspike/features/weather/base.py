"""Interface, registry and leakage-safe join for weather feature builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd

REGISTRY: dict[str, type[WeatherFeatureBuilder]] = {}


def register(cls: type[WeatherFeatureBuilder]) -> type[WeatherFeatureBuilder]:
    """Class decorator that makes a builder available by name in the config."""
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} must define a non-empty `name`")
    if cls.name in REGISTRY:
        raise ValueError(f"weather feature builder {cls.name!r} is already registered")
    REGISTRY[cls.name] = cls
    return cls


def with_availability(weather: pd.DataFrame, lag: pd.Timedelta) -> pd.DataFrame:
    """Add ``available_at = valid_time + lag``: the earliest time a value may be used.

    ``valid_time`` is already the time a value is complete (the end of the averaging hour for
    radiation; see :mod:`nemspike.data.weather`), so the lag models publication and processing
    delay.
    """
    if lag < pd.Timedelta(0):
        raise ValueError("availability lag must be non-negative")
    out = weather.sort_values("valid_time", ignore_index=True).copy()
    out["available_at"] = out["valid_time"] + lag
    return out


def asof_join(origins: pd.DatetimeIndex, table: pd.DataFrame) -> pd.DataFrame:
    """For each origin, take the latest row of ``table`` with ``available_at <= origin``.

    ``table`` must contain ``available_at``. Origins before any row is available get NaN.
    """
    left = pd.DataFrame({"origin": origins})
    right = table.sort_values("available_at")
    joined = pd.merge_asof(
        left.sort_values("origin"),
        right,
        left_on="origin",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    return joined.set_index("origin").reindex(origins)


class WeatherFeatureBuilder(ABC):
    """A family of weather features. Subclasses set ``name`` and implement ``build``."""

    name: ClassVar[str]

    @abstractmethod
    def build(self, weather: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
        """Return features indexed by ``origins``.

        ``weather`` has columns ``valid_time``, ``available_at`` and the raw variables. Use
        :func:`asof_join` to attach anything to origins.
        """


def build_weather_features(
    weather: pd.DataFrame,
    origins: pd.DatetimeIndex,
    builder_names: list[str],
    lag: pd.Timedelta,
) -> pd.DataFrame:
    """Run each named builder and concatenate the results (column names must not clash)."""
    unknown = set(builder_names) - set(REGISTRY)
    if unknown:
        raise ValueError(f"unknown weather builders {sorted(unknown)}; known: {sorted(REGISTRY)}")
    tagged = with_availability(weather, lag)
    frames = [REGISTRY[name]().build(tagged, origins) for name in builder_names]
    out = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=origins)
    if out.columns.has_duplicates:
        raise ValueError("weather builders produced duplicate column names")
    out.index.name = "origin"
    return out
