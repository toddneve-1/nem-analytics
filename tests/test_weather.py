from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nemspike.data import weather
from nemspike.data.weather import (
    VARIABLES,
    OpenMeteoError,
    WeatherQualityError,
    averaging_window_start,
    build_params,
    fetch_range,
    load_weather,
    parse_response,
    quality_report,
    utc_date_range,
)

VARS = list(VARIABLES)


def make_payload(start: str, end: str, drop_value_at: int | None = None) -> dict:
    """Synthetic Open-Meteo archive response (unixtime, UTC) for the given UTC dates."""
    times = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h", tz="UTC")
    n = len(times)
    rng = np.random.default_rng(0)
    hourly: dict[str, list] = {"time": [int(t.timestamp()) for t in times]}
    for v in VARS:
        lo, hi = VARIABLES[v].valid_range
        hourly[v] = list(rng.uniform(lo, min(hi, lo + 30), n).round(1))
    if drop_value_at is not None:
        hourly["temperature_2m"][drop_value_at] = None
    return {"latitude": -37.75, "longitude": 145.0, "utc_offset_seconds": 0, "hourly": hourly}


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, payload_override: dict | None = None) -> None:
        self.calls: list[dict] = []
        self.payload_override = payload_override

    def get(self, url: str, params: dict, timeout: float) -> FakeResponse:
        self.calls.append(params)
        if self.payload_override is not None:
            return FakeResponse(self.payload_override, status_code=429)
        return FakeResponse(make_payload(params["start_date"], params["end_date"]))


def test_build_params_requests_utc_unixtime_and_fixed_model() -> None:
    p = build_params(-37.8136, 144.9631, date(2024, 1, 1), date(2024, 12, 31), VARS, "era5")
    assert p["timezone"] == "GMT"
    assert p["timeformat"] == "unixtime"
    assert p["models"] == "era5"
    assert p["wind_speed_unit"] == "ms"


def test_build_params_rejects_unknown_variable() -> None:
    with pytest.raises(ValueError):
        build_params(0, 0, date(2024, 1, 1), date(2024, 1, 2), ["not_a_var"], "era5")


def test_parse_gives_utc_valid_time() -> None:
    df = parse_response(make_payload("2024-01-01", "2024-01-01"), VARS)
    assert len(df) == 24
    assert str(df["valid_time"].dt.tz) == "UTC"
    assert df["valid_time"].iloc[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")


def test_parse_keeps_missing_values_as_nan() -> None:
    df = parse_response(make_payload("2024-01-01", "2024-01-01", drop_value_at=5), VARS)
    assert np.isnan(df["temperature_2m"].iloc[5])
    assert quality_report(df, VARS).nan_counts["temperature_2m"] == 1


def test_parse_raises_on_api_error_payload() -> None:
    with pytest.raises(OpenMeteoError, match="limit"):
        parse_response({"error": True, "reason": "Daily API request limit exceeded"}, VARS)


def test_parse_rejects_non_utc_response() -> None:
    payload = make_payload("2024-01-01", "2024-01-01") | {"utc_offset_seconds": 36000}
    with pytest.raises(WeatherQualityError):
        parse_response(payload, VARS)


def test_parse_rejects_missing_variable() -> None:
    payload = make_payload("2024-01-01", "2024-01-01")
    del payload["hourly"]["wind_speed_100m"]
    with pytest.raises(WeatherQualityError):
        parse_response(payload, VARS)


def test_quality_report_flags_gap_and_out_of_range() -> None:
    df = parse_response(make_payload("2024-01-01", "2024-01-02"), VARS)
    df = df.drop(index=10).reset_index(drop=True)
    df.loc[0, "cloud_cover"] = 150.0
    report = quality_report(df, VARS)
    assert report.n_missing_hours == 1
    assert report.out_of_range["cloud_cover"] == 1


def test_radiation_label_is_end_of_averaging_hour() -> None:
    """Radiation labelled 12:00 describes 11:00-12:00: it must not be read as 12:00-13:00."""
    t = pd.Series([pd.Timestamp("2024-01-01 12:00", tz="UTC")])
    assert averaging_window_start("shortwave_radiation", t).iloc[0] == pd.Timestamp(
        "2024-01-01 11:00", tz="UTC"
    )
    assert averaging_window_start("temperature_2m", t).iloc[0] == t.iloc[0]


def test_utc_range_pads_nem_days() -> None:
    assert utc_date_range(date(2023, 1, 1), date(2026, 10, 31)) == (
        date(2022, 12, 31),
        date(2026, 11, 1),
    )


def test_fetch_range_caches_per_year_and_load_covers_nem_days(tmp_path: Path) -> None:
    session = FakeSession()
    kw = {"latitude": -37.8136, "longitude": 144.9631, "raw_dir": tmp_path}
    paths = fetch_range(date(2024, 1, 1), date(2024, 1, 31), session=session, **kw)  # type: ignore[arg-type]
    # NEM January 2024 needs UTC 31 Dec 2023 to 1 Feb 2024: two calendar years, two requests.
    assert len(paths) == 2 and len(session.calls) == 2
    fetch_range(date(2024, 1, 1), date(2024, 1, 31), session=session, **kw)  # type: ignore[arg-type]
    assert len(session.calls) == 2  # cached

    df = load_weather(date(2024, 1, 1), date(2024, 1, 31), **kw)
    first_nem_interval_start = pd.Timestamp("2023-12-31 14:00", tz="UTC")
    assert df["valid_time"].min() <= first_nem_interval_start
    assert df["valid_time"].is_monotonic_increasing


def test_rate_limit_is_raised_not_retried(tmp_path: Path) -> None:
    session = FakeSession({"error": True, "reason": "Daily API request limit exceeded."})
    with pytest.raises(OpenMeteoError):
        weather.fetch_year(
            2024,
            date(2024, 1, 1),
            date(2024, 1, 2),
            -37.8,
            144.9,
            tmp_path,
            session=session,  # type: ignore[arg-type]
        )
    assert len(session.calls) == 1


@pytest.mark.network
def test_live_fetch_two_days(tmp_path: Path) -> None:
    path = weather.fetch_year(
        2024, date(2024, 1, 1), date(2024, 1, 2), -37.8136, 144.9631, tmp_path
    )
    df = pd.read_parquet(path)
    assert len(df) == 48
