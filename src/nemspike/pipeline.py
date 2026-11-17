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
SNAPSHOT_DIR = Path("results") / "predictions"


def load_predictions(stage: str, horizon_minutes: int, model: str) -> pd.DataFrame:
    """Out-of-sample backtest predictions for one horizon and model.

    Prefers the full predictions written by ``nemspike backtest`` (git-ignored ``artefacts/``).
    If they are absent, falls back to the compact snapshot committed under ``results/predictions``
    (32-bit floats, so downstream results can differ from a full run in the last digits).
    """
    full = ARTEFACT_DIR / "backtest" / stage / f"h{horizon_minutes}" / f"{model}.parquet"
    snapshot = SNAPSHOT_DIR / stage / f"h{horizon_minutes}_{model}.parquet"
    for path in (full, snapshot):
        if path.exists():
            return pd.read_parquet(path)
    raise FileNotFoundError(
        f"no {stage} predictions for {model} at {horizon_minutes} minutes (looked for {full} and "
        f"{snapshot}). Run: nemspike backtest --models {model} --horizons {horizon_minutes}"
    )


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached prices and weather for the configured period."""
    d, w = cfg.data, cfg.data.weather
    prices = load_price_demand(d.start, d.end, d.region, d.raw_dir)
    weather = load_weather(d.start, d.end, w.latitude, w.longitude, d.raw_dir, model=w.model)
    return prices, weather


def build_horizon(cfg: Config, horizon_minutes: int) -> pd.DataFrame:
    prices, weather = load_inputs(cfg)
    return build_features(prices, weather, horizon_minutes, cfg)
