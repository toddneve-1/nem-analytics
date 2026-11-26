"""Calendar and solar-geometry features of the *target* interval.

These are deterministic: the hour, weekday, public holidays and the sun's position at a future
time are known in advance, so using them for the target time is not leakage. Calendar fields use
NEM time (AEST), because market behaviour follows the market clock.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from nemspike.timeutils import INTERVAL, to_nem_time


def solar_elevation_deg(
    times_utc: pd.DatetimeIndex, latitude: float, longitude: float
) -> np.ndarray:
    """Approximate solar elevation angle (degrees) using NOAA's general solar position equations.

    Accurate to roughly a degree, which is ample for a feature. Negative values mean the sun is
    below the horizon.
    """
    t = times_utc.tz_convert("UTC")
    doy = t.day_of_year.to_numpy()
    hour = (t.hour + t.minute / 60 + t.second / 3600).to_numpy()
    gamma = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    true_solar_minutes = hour * 60 + eqtime + 4 * longitude
    hour_angle = np.deg2rad(true_solar_minutes / 4 - 180)
    lat = np.deg2rad(latitude)
    cos_zenith = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return 90 - np.rad2deg(np.arccos(np.clip(cos_zenith, -1, 1)))


def calendar_features(
    origins: pd.DatetimeIndex, horizon_minutes: int, latitude: float, longitude: float
) -> pd.DataFrame:
    """Calendar and sun-position features for the target interval ending at origin + horizon."""
    target_end = origins + pd.Timedelta(minutes=horizon_minutes)
    # Label calendar fields by the interval start, so the interval 23:55-24:00 belongs to the
    # day it falls in rather than to the next day.
    target_start_nem = to_nem_time(target_end - INTERVAL)
    years = sorted(set(target_start_nem.year))
    vic_holidays = holidays.Australia(subdiv="VIC", years=years)
    dates = target_start_nem.date
    midpoint = target_end - INTERVAL / 2

    df = pd.DataFrame(
        {
            "target_hour": target_start_nem.hour,
            "target_minute_of_day": target_start_nem.hour * 60 + target_start_nem.minute,
            "target_dayofweek": target_start_nem.day_of_week,
            "target_month": target_start_nem.month,
            "target_dayofyear": target_start_nem.day_of_year,
            "target_is_weekend": (target_start_nem.day_of_week >= 5).astype(int),
            "target_is_holiday": np.array([d in vic_holidays for d in dates], dtype=int),
            "target_solar_elevation": solar_elevation_deg(midpoint, latitude, longitude),
        },
        index=origins,
    )
    df.index.name = "origin"
    return df
