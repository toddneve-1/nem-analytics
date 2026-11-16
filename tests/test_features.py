from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from nemspike.config import Config
from nemspike.features.build import TARGET, build_features, feature_columns, make_target
from nemspike.features.calendar import calendar_features, solar_elevation_deg
from nemspike.features.leakage import find_lookahead
from nemspike.features.price import seasonal_features, to_regular_grid
from nemspike.features.weather import REGISTRY, WeatherFeatureBuilder, asof_join, register
from nemspike.features.weather.base import build_weather_features
from nemspike.features.weather.basic import wind_components

HORIZONS = [30, 240]


def price_at(prices: pd.DataFrame, t: pd.Timestamp) -> float:
    return float(prices.set_index("interval_end").loc[t, "rrp"])


# --- targets and price features -------------------------------------------------------------


@pytest.mark.parametrize("h", HORIZONS)
def test_target_is_price_at_origin_plus_horizon(prices, weather, cfg: Config, h: int) -> None:
    df = build_features(prices, weather, h, cfg)
    origin = df.index[500]
    assert df.loc[origin, TARGET] == price_at(prices, origin + pd.Timedelta(minutes=h))
    assert df.loc[origin, "target_time"] == origin + pd.Timedelta(minutes=h)


def test_target_is_nan_beyond_data(prices) -> None:
    grid = to_regular_grid(prices)
    target = make_target(grid, 30)
    assert target.iloc[-6:].isna().all() and target.iloc[:-6].notna().all()


def test_lag_zero_is_price_at_origin(prices, weather, cfg: Config) -> None:
    df = build_features(prices, weather, 30, cfg)
    origin = df.index[1000]
    assert df.loc[origin, "rrp_lag_0"] == price_at(prices, origin)
    assert df.loc[origin, "rrp_lag_12"] == price_at(prices, origin - pd.Timedelta(hours=1))


def test_rolling_window_ends_at_origin(prices, weather, cfg: Config) -> None:
    df = build_features(prices, weather, 30, cfg)
    origin = df.index[1000]
    s = prices.set_index("interval_end")["rrp"]
    window = s[(s.index > origin - pd.Timedelta(minutes=60)) & (s.index <= origin)]
    assert len(window) == 12
    assert df.loc[origin, "rrp_max_60m"] == window.max()


@pytest.mark.parametrize("h", HORIZONS)
def test_seasonal_lags_refer_to_target_time(prices, h: int) -> None:
    grid = to_regular_grid(prices)
    sf = seasonal_features(grid, h)
    origin = grid.index[3000]
    target_time = origin + pd.Timedelta(minutes=h)
    assert sf.loc[origin, "rrp_same_time_yesterday"] == price_at(
        prices, target_time - pd.Timedelta(days=1)
    )
    assert sf.loc[origin, "rrp_same_time_last_week"] == price_at(
        prices, target_time - pd.Timedelta(days=7)
    )


def test_seasonal_rejects_horizon_beyond_a_day(prices) -> None:
    with pytest.raises(ValueError):
        seasonal_features(to_regular_grid(prices), 24 * 60 + 5)


def test_missing_interval_is_not_filled(prices, cfg: Config) -> None:
    gappy = prices.drop(index=3000)
    grid = to_regular_grid(gappy)
    assert grid["rrp"].isna().sum() == 1


# --- calendar features -----------------------------------------------------------------------


def test_calendar_uses_nem_time_and_vic_holidays() -> None:
    # Origin 2024-11-04 23:30 UTC = 2024-11-05 09:30 AEST (Melbourne Cup Day); 30-min horizon.
    origins = pd.DatetimeIndex([pd.Timestamp("2024-11-04 23:30", tz="UTC")])
    cal = calendar_features(origins, 30, -37.81, 144.96)
    row = cal.iloc[0]
    assert row["target_hour"] == 9 and row["target_minute_of_day"] == 9 * 60 + 55
    assert row["target_is_holiday"] == 1
    assert row["target_dayofweek"] == 1  # Tuesday


def test_interval_ending_midnight_belongs_to_previous_day() -> None:
    # Target interval 23:55-24:00 AEST on Sunday 7 Jan 2024 ends at 2024-01-07 14:00 UTC.
    origins = pd.DatetimeIndex([pd.Timestamp("2024-01-07 13:55", tz="UTC")])
    row = calendar_features(origins, 5, -37.81, 144.96).iloc[0]
    assert row["target_dayofweek"] == 6 and row["target_hour"] == 23


def test_solar_elevation_reasonable_for_melbourne() -> None:
    # Local solar noon near the December solstice is about 02:20 UTC; elevation ≈ 90 - 37.8 + 23.4.
    noon = pd.DatetimeIndex([pd.Timestamp("2024-12-21 02:20", tz="UTC")])
    midnight = pd.DatetimeIndex([pd.Timestamp("2024-12-21 14:20", tz="UTC")])
    assert solar_elevation_deg(noon, -37.81, 144.96)[0] == pytest.approx(75.6, abs=1.5)
    assert solar_elevation_deg(midnight, -37.81, 144.96)[0] < -20


# --- weather features ------------------------------------------------------------------------


def test_weather_respects_availability_lag(prices, weather, cfg: Config) -> None:
    df = build_features(prices, weather, 30, cfg)
    lag = pd.Timedelta(hours=cfg.data.weather.availability_lag_hours)
    origin = pd.Timestamp("2024-01-10 05:35", tz="UTC")
    # Latest usable valid_time is floor(origin - lag) to the hour = 04:00.
    expected = weather.set_index("valid_time").loc[(origin - lag).floor("h"), "temperature_2m"]
    assert df.loc[origin, "wx_temperature"] == expected
    assert df["wx_age_minutes"].dropna().min() >= lag.total_seconds() / 60


def test_radiation_for_hour_ending_noon_first_usable_at_one_pm(weather) -> None:
    """Radiation labelled 12:00 averages 11:00-12:00; with a 1 h lag it is first usable at 13:00."""
    lag = pd.Timedelta(hours=1)
    origins = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-10 12:55", tz="UTC"), pd.Timestamp("2024-01-10 13:00", tz="UTC")]
    )
    f = build_weather_features(weather, origins, ["basic"], lag)
    noon_value = weather.set_index("valid_time").loc["2024-01-10 12:00", "shortwave_radiation"]
    assert f.iloc[0]["wx_shortwave_radiation"] != noon_value
    assert f.iloc[1]["wx_shortwave_radiation"] == noon_value


def test_asof_join_never_uses_future_rows() -> None:
    table = pd.DataFrame(
        {
            "available_at": pd.to_datetime(["2024-01-01 01:00", "2024-01-01 02:00"], utc=True),
            "x": [1.0, 2.0],
        }
    )
    origins = pd.to_datetime(["2024-01-01 00:30", "2024-01-01 01:59", "2024-01-01 02:00"], utc=True)
    out = asof_join(pd.DatetimeIndex(origins), table)
    assert np.isnan(out["x"].iloc[0])
    assert out["x"].tolist()[1:] == [1.0, 2.0]


def test_wind_components_follow_meteorological_convention() -> None:
    u, v = wind_components(pd.Series([10.0, 10.0]), pd.Series([0.0, 270.0]))
    # Northerly (from north) blows south: v = -10. Westerly (from west) blows east: u = +10.
    assert v.iloc[0] == pytest.approx(-10) and u.iloc[0] == pytest.approx(0, abs=1e-9)
    assert u.iloc[1] == pytest.approx(10) and v.iloc[1] == pytest.approx(0, abs=1e-9)


def test_unknown_weather_builder_raises(weather) -> None:
    with pytest.raises(ValueError, match="unknown weather builders"):
        build_weather_features(weather, pd.DatetimeIndex([]), ["nope"], pd.Timedelta(hours=1))


def test_custom_builder_plugs_in(weather) -> None:
    @register
    class _TempOnly(WeatherFeatureBuilder):
        name: ClassVar[str] = "_test_temp_only"

        def build(self, weather: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
            return asof_join(origins, weather[["available_at", "temperature_2m"]]).rename(
                columns={"temperature_2m": "t_custom"}
            )[["t_custom"]]

    try:
        origins = pd.DatetimeIndex([pd.Timestamp("2024-01-10 06:00", tz="UTC")])
        out = build_weather_features(
            weather, origins, ["basic", "_test_temp_only"], pd.Timedelta(hours=1)
        )
        assert out.loc[origins[0], "t_custom"] == out.loc[origins[0], "wx_temperature"]
    finally:
        REGISTRY.pop("_test_temp_only")


# --- end-to-end leakage check ----------------------------------------------------------------


def _sample_origins(df: pd.DataFrame, n: int = 6) -> list[pd.Timestamp]:
    idx = np.linspace(100, len(df) - 400, n).astype(int)
    return [df.index[i] for i in idx]


@pytest.mark.parametrize("h", HORIZONS)
def test_no_lookahead_in_any_feature(prices, weather, cfg: Config, h: int) -> None:
    def build(p: pd.DataFrame, w: pd.DataFrame | None) -> pd.DataFrame:
        return build_features(p, w, h, cfg)

    origins = _sample_origins(build(prices, weather))
    lag = pd.Timedelta(hours=cfg.data.weather.availability_lag_hours)
    assert find_lookahead(build, prices, weather, origins, lag) == {}


def test_lookahead_check_catches_a_deliberate_leak(prices, weather, cfg: Config) -> None:
    """Test the test: a feature that peeks one interval ahead must be reported."""

    def leaky_build(p: pd.DataFrame, w: pd.DataFrame | None) -> pd.DataFrame:
        df = build_features(p, w, 30, cfg)
        df["leaky_next_price"] = df["rrp_lag_0"].shift(-1)
        return df

    origins = _sample_origins(leaky_build(prices, weather), n=3)
    lag = pd.Timedelta(hours=cfg.data.weather.availability_lag_hours)
    leaks = find_lookahead(leaky_build, prices, weather, origins, lag)
    assert set(leaks) == set(origins)
    assert all(cols == ["leaky_next_price"] for cols in leaks.values())


def test_weather_leak_is_caught_when_lag_is_ignored(prices, weather, cfg: Config) -> None:
    """If features used weather with zero lag while we claim a 1 h lag, the check must fail."""
    zero_lag_cfg = cfg.model_copy(deep=True)
    zero_lag_cfg.data.weather.availability_lag_hours = 0

    def build(p: pd.DataFrame, w: pd.DataFrame | None) -> pd.DataFrame:
        return build_features(p, w, 30, zero_lag_cfg)

    # Origins on the hour, where a zero-lag build uses the value labelled at the origin itself.
    origins = [
        pd.Timestamp("2024-01-10 06:00", tz="UTC"),
        pd.Timestamp("2024-01-12 18:00", tz="UTC"),
    ]
    leaks = find_lookahead(build, prices, weather, origins, pd.Timedelta(hours=1))
    assert set(leaks) == set(origins)
    assert all(any(c.startswith("wx_") for c in cols) for cols in leaks.values())


def test_feature_columns_exclude_target(prices, weather, cfg: Config) -> None:
    cols = feature_columns(build_features(prices, weather, 30, cfg))
    assert TARGET not in cols and "target_time" not in cols
    assert any(c.startswith("wx_") for c in cols)
