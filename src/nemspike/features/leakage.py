"""Lookahead (leakage) check by future perturbation.

Idea: a feature at origin ``T`` must not change if we replace everything that happened after
``T`` with garbage. For each sampled origin we:

1. build features from the true data;
2. replace all prices/demand for intervals ending after ``T``, and all weather that becomes
   available after ``T``, with random values;
3. rebuild and require the feature row at ``T`` to be identical.

This tests the *whole* pipeline end to end, so it also catches leaks introduced later by new
features or new weather builders, not just the ones we thought of.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from nemspike.features.build import feature_columns

BuildFn = Callable[[pd.DataFrame, pd.DataFrame | None], pd.DataFrame]


def _perturb_after(
    prices: pd.DataFrame,
    weather: pd.DataFrame | None,
    origin: pd.Timestamp,
    weather_lag: pd.Timedelta,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    p = prices.copy()
    future = p["interval_end"] > origin
    p.loc[future, "rrp"] = rng.normal(5000, 3000, future.sum())
    p.loc[future, "demand_mw"] = rng.normal(9000, 2000, future.sum())
    if weather is None:
        return p, None
    w = weather.copy()
    future_w = (w["valid_time"] + weather_lag) > origin
    for col in w.columns.drop("valid_time"):
        w.loc[future_w, col] = rng.uniform(0, 100, future_w.sum())
    return p, w


def find_lookahead(
    build: BuildFn,
    prices: pd.DataFrame,
    weather: pd.DataFrame | None,
    origins: Iterable[pd.Timestamp],
    weather_lag: pd.Timedelta,
    seed: int = 0,
) -> dict[pd.Timestamp, list[str]]:
    """Return ``{origin: [features that changed]}``; an empty dict means no leakage found."""
    rng = np.random.default_rng(seed)
    baseline = build(prices, weather)
    cols = feature_columns(baseline)
    leaks: dict[pd.Timestamp, list[str]] = {}
    for origin in origins:
        p, w = _perturb_after(prices, weather, origin, weather_lag, rng)
        perturbed = build(p, w)
        a = baseline.loc[origin, cols]
        b = perturbed.loc[origin, cols]
        same = (a == b) | (a.isna() & b.isna())
        if not same.all():
            leaks[origin] = sorted(same.index[~same])
    return leaks
