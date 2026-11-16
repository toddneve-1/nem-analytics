"""Basic weather features: the latest available values and their recent changes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nemspike.features.weather.base import WeatherFeatureBuilder, asof_join, register

HOUR = pd.Timedelta(hours=1)


def wind_components(speed: pd.Series, direction_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert meteorological wind (speed, direction blown FROM) to u (eastward) and v (northward).

    Direction is circular (359° is next to 1°), so raw degrees mislead a model. A northerly
    (from 0°) blows towards the south, giving v < 0.
    """
    rad = np.deg2rad(direction_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


@register
class BasicWeatherFeatures(WeatherFeatureBuilder):
    """Latest observed temperature, wind, cloud and radiation, plus 3-hour changes.

    Changes are computed on the hourly weather grid *before* the as-of join, so each one only
    uses values that were themselves available by the time the joined row was available.
    """

    name = "basic"

    def build(self, weather: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
        w = weather.set_index("valid_time").sort_index()
        full = pd.date_range(w.index.min(), w.index.max(), freq=HOUR, name="valid_time")
        w = w.reindex(full)
        w["available_at"] = w.index + (weather["available_at"] - weather["valid_time"]).iloc[0]

        f = pd.DataFrame(index=w.index)
        f["wx_temperature"] = w["temperature_2m"]
        f["wx_humidity"] = w["relative_humidity_2m"]
        f["wx_cloud_cover"] = w["cloud_cover"]
        f["wx_wind_speed_10m"] = w["wind_speed_10m"]
        f["wx_wind_speed_100m"] = w["wind_speed_100m"]
        f["wx_wind_gusts_10m"] = w["wind_gusts_10m"]
        f["wx_wind_u_100m"], f["wx_wind_v_100m"] = wind_components(
            w["wind_speed_100m"], w["wind_direction_100m"]
        )
        f["wx_wind_u_10m"], f["wx_wind_v_10m"] = wind_components(
            w["wind_speed_10m"], w["wind_direction_10m"]
        )
        f["wx_shortwave_radiation"] = w["shortwave_radiation"]
        f["wx_temperature_change_3h"] = w["temperature_2m"] - w["temperature_2m"].shift(3)
        f["wx_wind_speed_100m_change_3h"] = w["wind_speed_100m"] - w["wind_speed_100m"].shift(3)
        f["wx_temperature_max_24h"] = w["temperature_2m"].rolling(24, min_periods=18).max()
        f["available_at"] = w["available_at"]
        f["wx_valid_time"] = w.index

        joined = asof_join(origins, f.reset_index(drop=True))
        # How old the latest weather is at the origin, in minutes (also a useful sanity check).
        joined["wx_age_minutes"] = (
            joined.index.to_series() - joined["wx_valid_time"]
        ).dt.total_seconds().to_numpy() / 60
        return joined.drop(columns=["available_at", "wx_valid_time"])
