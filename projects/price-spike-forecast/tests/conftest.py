"""Shared synthetic fixtures. No network access."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nemspike.config import Config, load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config(DEFAULT_CONFIG)


def make_prices(days: int = 21, seed: int = 0) -> pd.DataFrame:
    """Synthetic 5-minute prices with a daily cycle, negative middays and occasional spikes."""
    rng = np.random.default_rng(seed)
    ends = pd.date_range("2024-01-01 14:05", periods=days * 288, freq="5min", tz="UTC")
    hour_nem = ((ends.hour + 10) % 24).to_numpy()
    base = (
        60
        + 40 * np.sin((hour_nem - 12) / 24 * 2 * np.pi)
        - 70 * ((hour_nem >= 10) & (hour_nem <= 14))
    )
    rrp = base + rng.normal(0, 15, len(ends))
    spikes = rng.random(len(ends)) < 0.01
    rrp[spikes] = rng.uniform(300, 15000, spikes.sum())
    demand = 4500 + 800 * np.sin((hour_nem - 6) / 24 * 2 * np.pi) + rng.normal(0, 100, len(ends))
    return pd.DataFrame(
        {
            "interval_start": ends - pd.Timedelta(minutes=5),
            "interval_end": ends,
            "region": "VIC1",
            "rrp": rrp,
            "demand_mw": demand,
            "period_type": "TRADE",
        }
    )


def make_weather(prices: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Synthetic hourly weather covering the price period plus a day either side."""
    rng = np.random.default_rng(seed)
    start = prices["interval_end"].min().floor("D") - pd.Timedelta(days=1)
    end = prices["interval_end"].max().ceil("D") + pd.Timedelta(days=1)
    t = pd.date_range(start, end, freq="h", tz="UTC")
    n = len(t)
    return pd.DataFrame(
        {
            "valid_time": t,
            "temperature_2m": 20 + 8 * np.sin(np.arange(n) / 24 * 2 * np.pi) + rng.normal(0, 1, n),
            "relative_humidity_2m": rng.uniform(30, 90, n),
            "cloud_cover": rng.uniform(0, 100, n),
            "wind_speed_10m": rng.uniform(0, 12, n),
            "wind_speed_100m": rng.uniform(0, 20, n),
            "wind_direction_10m": rng.uniform(0, 360, n),
            "wind_direction_100m": rng.uniform(0, 360, n),
            "wind_gusts_10m": rng.uniform(0, 25, n),
            "shortwave_radiation": rng.uniform(0, 1000, n),
            "direct_radiation": rng.uniform(0, 800, n),
            "diffuse_radiation": rng.uniform(0, 300, n),
        }
    )


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture(scope="session")
def weather(prices: pd.DataFrame) -> pd.DataFrame:
    return make_weather(prices)
