"""Shared steps used by several CLI commands: load cached inputs and build a horizon's table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nemspike.config import Config
from nemspike.data.aemo import load_price_demand
from nemspike.data.weather import load_weather
from nemspike.features.build import build_features

ARTEFACT_DIR = Path("artefacts")
REPORT_DIR = Path("reports")


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached prices and weather for the configured period."""
    d, w = cfg.data, cfg.data.weather
    prices = load_price_demand(d.start, d.end, d.region, d.raw_dir)
    weather = load_weather(d.start, d.end, w.latitude, w.longitude, d.raw_dir, model=w.model)
    return prices, weather


def build_horizon(cfg: Config, horizon_minutes: int) -> pd.DataFrame:
    prices, weather = load_inputs(cfg)
    return build_features(prices, weather, horizon_minutes, cfg)
