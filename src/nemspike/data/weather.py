"""Historical hourly weather from the Open-Meteo Historical Weather API (ERA5 reanalysis).

API: ``https://archive-api.open-meteo.com/v1/archive``. Data licence: CC BY 4.0. The free API is
for non-commercial use only.

Time conventions (from Open-Meteo's documentation), which matter for leakage:

* **Instant** variables (temperature, wind, cloud cover) are values *at* the labelled hour.
* **Preceding-hour mean** variables (radiation) are averages over the hour *ending* at the label:
  radiation labelled 12:00 covers 11:00-12:00.

In both cases the value is complete at its label time, so we store the label as ``valid_time``
(UTC). Feature construction then adds an explicit availability lag (config
``data.weather.availability_lag_hours``) before a value may be used at a forecast origin.

We request ``timeformat=unixtime`` so timestamps are unambiguous UTC seconds. Open-Meteo's ISO
format applies a single fixed offset to the whole response and ignores daylight-saving changes.

We request ``models=era5`` explicitly. The default "best match" blends several models whose mix
can change over time, which would add artificial jumps to the features.

Limitation: ERA5 is a reanalysis published about 5 days after real time. It stands in here for
real-time observations that a trader would have had (for example Bureau of Meteorology automatic
weather stations), and it is smoother than a point observation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class Aggregation(StrEnum):
    INSTANT = "instant"
    PRECEDING_HOUR_MEAN = "preceding_hour_mean"


@dataclass(frozen=True)
class VariableSpec:
    aggregation: Aggregation
    unit: str
    valid_range: tuple[float, float]


# Variables fetched by default. Units assume wind_speed_unit=ms.
VARIABLES: dict[str, VariableSpec] = {
    "temperature_2m": VariableSpec(Aggregation.INSTANT, "°C", (-30.0, 55.0)),
    "relative_humidity_2m": VariableSpec(Aggregation.INSTANT, "%", (0.0, 100.0)),
    "cloud_cover": VariableSpec(Aggregation.INSTANT, "%", (0.0, 100.0)),
    "wind_speed_10m": VariableSpec(Aggregation.INSTANT, "m/s", (0.0, 75.0)),
    "wind_speed_100m": VariableSpec(Aggregation.INSTANT, "m/s", (0.0, 100.0)),
    "wind_direction_10m": VariableSpec(Aggregation.INSTANT, "°", (0.0, 360.0)),
    "wind_direction_100m": VariableSpec(Aggregation.INSTANT, "°", (0.0, 360.0)),
    "wind_gusts_10m": VariableSpec(Aggregation.INSTANT, "m/s", (0.0, 100.0)),
    "shortwave_radiation": VariableSpec(Aggregation.PRECEDING_HOUR_MEAN, "W/m²", (0.0, 1500.0)),
    "direct_radiation": VariableSpec(Aggregation.PRECEDING_HOUR_MEAN, "W/m²", (0.0, 1500.0)),
    "diffuse_radiation": VariableSpec(Aggregation.PRECEDING_HOUR_MEAN, "W/m²", (0.0, 1000.0)),
}

HOUR = pd.Timedelta(hours=1)


class OpenMeteoError(RuntimeError):
    """The API returned an error payload (for example a rate-limit message)."""


class WeatherQualityError(ValueError):
    """Downloaded weather data cannot be used safely."""


def build_params(
    latitude: float, longitude: float, start: date, end: date, variables: list[str], model: str
) -> dict[str, str]:
    """Query parameters for one archive request (``start``..``end`` are UTC dates)."""
    unknown = set(variables) - set(VARIABLES)
    if unknown:
        raise ValueError(f"unknown weather variables: {sorted(unknown)}")
    return {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(variables),
        "models": model,
        "timezone": "GMT",
        "timeformat": "unixtime",
        "wind_speed_unit": "ms",
    }


def parse_response(payload: dict, variables: list[str]) -> pd.DataFrame:
    """Turn an Open-Meteo JSON payload into a DataFrame indexed by UTC ``valid_time``.

    Missing values are kept as NaN and reported by :func:`quality_report`; they are not filled.
    """
    if payload.get("error"):
        raise OpenMeteoError(payload.get("reason", "unknown Open-Meteo error"))
    if payload.get("utc_offset_seconds", 0) != 0:
        raise WeatherQualityError("expected UTC timestamps (utc_offset_seconds == 0)")
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise WeatherQualityError("response has no hourly data")
    missing = [v for v in variables if v not in hourly]
    if missing:
        raise WeatherQualityError(f"response is missing variables: {missing}")

    df = pd.DataFrame(
        {v: pd.to_numeric(pd.Series(hourly[v], dtype="object"), errors="coerce") for v in variables}
    ).astype("float64")
    df.insert(0, "valid_time", pd.to_datetime(hourly["time"], unit="s", utc=True))
    if df["valid_time"].duplicated().any():
        raise WeatherQualityError("duplicated timestamps in weather response")
    return df.sort_values("valid_time", ignore_index=True)


@dataclass
class WeatherQualityReport:
    n_rows: int
    n_expected: int
    n_missing_hours: int
    nan_counts: dict[str, int]
    out_of_range: dict[str, int]

    def summary(self) -> str:
        nans = {k: v for k, v in self.nan_counts.items() if v}
        oor = {k: v for k, v in self.out_of_range.items() if v}
        return (
            f"rows {self.n_rows:,} of {self.n_expected:,} expected hours; "
            f"missing hours {self.n_missing_hours:,}; NaNs {nans or 'none'}; "
            f"out of range {oor or 'none'}"
        )


def quality_report(df: pd.DataFrame, variables: list[str]) -> WeatherQualityReport:
    """Check hourly continuity, missing values and physically implausible values."""
    if df.empty:
        return WeatherQualityReport(0, 0, 0, {}, {})
    expected = pd.date_range(df["valid_time"].min(), df["valid_time"].max(), freq=HOUR)
    present = pd.DatetimeIndex(df["valid_time"])
    out_of_range = {}
    for v in variables:
        lo, hi = VARIABLES[v].valid_range
        s = df[v].dropna()
        out_of_range[v] = int(((s < lo) | (s > hi)).sum())
    return WeatherQualityReport(
        n_rows=len(df),
        n_expected=len(expected),
        n_missing_hours=len(expected.difference(present)),
        nan_counts={v: int(df[v].isna().sum()) for v in variables},
        out_of_range=out_of_range,
    )


def averaging_window_start(variable: str, valid_time: pd.Series) -> pd.Series:
    """Start of the period a value describes: the label itself for instant variables, and one
    hour earlier for preceding-hour means."""
    if VARIABLES[variable].aggregation is Aggregation.PRECEDING_HOUR_MEAN:
        return valid_time - HOUR
    return valid_time


def _cache_path(raw_dir: Path, model: str, lat: float, lon: float, year: int) -> Path:
    loc = f"{lat:.4f}_{lon:.4f}".replace("-", "m")
    return raw_dir / "weather" / f"openmeteo_{model}" / loc / f"{year}.parquet"


def _get_json(
    params: dict[str, str], session: requests.Session, retries: int = 3, timeout: float = 120
) -> dict:
    """GET with backoff on network errors and 5xx. API error payloads are raised, not retried,
    because a rate-limit message will not clear by retrying immediately."""
    for attempt in range(retries + 1):
        try:
            resp = session.get(ARCHIVE_URL, params=params, timeout=timeout)
            if resp.status_code < 500:
                payload = resp.json()
                if payload.get("error"):
                    raise OpenMeteoError(payload.get("reason", f"HTTP {resp.status_code}"))
                resp.raise_for_status()
                return payload
            log.warning("HTTP %s from Open-Meteo", resp.status_code)
        except requests.ConnectionError as exc:
            log.warning("connection error from Open-Meteo: %s", exc)
        if attempt < retries:
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch Open-Meteo data after {retries + 1} attempts")


def fetch_year(
    year: int,
    start: date,
    end: date,
    latitude: float,
    longitude: float,
    raw_dir: Path,
    variables: list[str] | None = None,
    model: str = "era5",
    session: requests.Session | None = None,
    force: bool = False,
) -> Path:
    """Fetch one calendar year (UTC), clipped to ``start``..``end``, and cache it as parquet."""
    variables = variables or list(VARIABLES)
    path = _cache_path(raw_dir, model, latitude, longitude, year)
    if path.exists() and not force:
        return path
    y_start = max(start, date(year, 1, 1))
    y_end = min(end, date(year, 12, 31))
    params = build_params(latitude, longitude, y_start, y_end, variables, model)
    log.info("fetching Open-Meteo %s %s to %s", model, y_start, y_end)
    df = parse_response(_get_json(params, session or requests.Session()), variables)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def utc_date_range(nem_start: date, nem_end: date) -> tuple[date, date]:
    """UTC dates covering NEM days ``nem_start``..``nem_end``, padded by one day either side.

    NEM days start at 14:00 UTC the previous day, and lagged features need some history.
    """
    return nem_start - timedelta(days=1), nem_end + timedelta(days=1)


def fetch_range(
    nem_start: date,
    nem_end: date,
    latitude: float,
    longitude: float,
    raw_dir: Path,
    variables: list[str] | None = None,
    model: str = "era5",
    session: requests.Session | None = None,
    force: bool = False,
) -> list[Path]:
    """Fetch every calendar year needed to cover NEM days ``nem_start``..``nem_end``."""
    start, end = utc_date_range(nem_start, nem_end)
    session = session or requests.Session()
    return [
        fetch_year(
            y, start, end, latitude, longitude, raw_dir, variables, model, session, force=force
        )
        for y in range(start.year, end.year + 1)
    ]


def load_weather(
    nem_start: date,
    nem_end: date,
    latitude: float,
    longitude: float,
    raw_dir: Path,
    variables: list[str] | None = None,
    model: str = "era5",
) -> pd.DataFrame:
    """Load cached years covering NEM days ``nem_start``..``nem_end`` (with one day of padding)."""
    variables = variables or list(VARIABLES)
    start, end = utc_date_range(nem_start, nem_end)
    frames = []
    for y in range(start.year, end.year + 1):
        path = _cache_path(raw_dir, model, latitude, longitude, y)
        if not path.exists():
            raise FileNotFoundError(f"{path} not cached; run `nemspike fetch-weather` first")
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)
    df = df[(df["valid_time"] >= lo) & (df["valid_time"] <= hi)]
    if df["valid_time"].duplicated().any():
        raise WeatherQualityError("duplicated timestamps across cached years")
    return df.sort_values("valid_time", ignore_index=True)[["valid_time", *variables]]
