"""Price and demand history features, indexed by forecast origin.

All features are computed on a regular 5-minute grid so that ``shift(k)`` always means "k
intervals earlier". Missing intervals stay NaN; nothing is forward-filled.
"""

from __future__ import annotations

import pandas as pd

from nemspike.config import FeaturesConfig
from nemspike.timeutils import INTERVAL

INTERVALS_PER_DAY = 288
INTERVALS_PER_WEEK = 7 * INTERVALS_PER_DAY


def to_regular_grid(prices: pd.DataFrame) -> pd.DataFrame:
    """Index price/demand rows by ``interval_end`` on a complete 5-minute grid (gaps -> NaN)."""
    s = prices.set_index("interval_end")[["rrp", "demand_mw"]].sort_index()
    if s.index.has_duplicates:
        raise ValueError("duplicated interval_end values")
    full = pd.date_range(s.index.min(), s.index.max(), freq=INTERVAL, name="origin")
    return s.reindex(full)


def _intervals(minutes: int) -> int:
    if minutes % 5:
        raise ValueError(f"{minutes} is not a multiple of 5 minutes")
    return minutes // 5


def history_features(grid: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Horizon-independent features from prices and demand up to and including the origin."""
    rrp, demand = grid["rrp"], grid["demand_mw"]
    out: dict[str, pd.Series] = {}

    for k in cfg.price_lags_intervals:
        out[f"rrp_lag_{k}"] = rrp.shift(k)
        out[f"demand_lag_{k}"] = demand.shift(k)

    out["rrp_change_1"] = rrp - rrp.shift(1)
    out["rrp_change_6"] = rrp - rrp.shift(6)
    out["demand_change_6"] = demand - demand.shift(6)
    out["demand_change_12"] = demand - demand.shift(12)

    for w in cfg.rolling_windows_minutes:
        n = _intervals(w)
        roll = rrp.rolling(n, min_periods=n)
        out[f"rrp_mean_{w}m"] = roll.mean()
        out[f"rrp_std_{w}m"] = roll.std()
        out[f"rrp_min_{w}m"] = roll.min()
        out[f"rrp_max_{w}m"] = roll.max()
        out[f"rrp_median_{w}m"] = roll.median()
        out[f"rrp_neg_share_{w}m"] = (rrp < 0).astype(float).where(rrp.notna()).rolling(n, n).mean()
        out[f"demand_mean_{w}m"] = demand.rolling(n, min_periods=n).mean()
        out[f"demand_max_{w}m"] = demand.rolling(n, min_periods=n).max()
        for t in cfg.count_above_thresholds:
            above = (rrp > t).astype(float).where(rrp.notna())
            out[f"rrp_count_above_{t:g}_{w}m"] = above.rolling(n, min_periods=n).sum()

    df = pd.DataFrame(out, index=grid.index)
    df.index.name = "origin"
    return df


def seasonal_features(grid: pd.DataFrame, horizon_minutes: int) -> pd.DataFrame:
    """Price and demand at the same time of day as the *target*, one day and one week earlier.

    For origin T and horizon h the target interval ends at T + h. "Same time yesterday" is the
    interval ending at T + h - 24 h, which is at or before T only if h <= 24 h.
    """
    h = _intervals(horizon_minutes)
    if h > INTERVALS_PER_DAY:
        raise ValueError("seasonal features require a horizon of at most 24 hours")
    rrp, demand = grid["rrp"], grid["demand_mw"]
    df = pd.DataFrame(
        {
            "rrp_same_time_yesterday": rrp.shift(INTERVALS_PER_DAY - h),
            "rrp_same_time_last_week": rrp.shift(INTERVALS_PER_WEEK - h),
            "demand_same_time_yesterday": demand.shift(INTERVALS_PER_DAY - h),
            "demand_same_time_last_week": demand.shift(INTERVALS_PER_WEEK - h),
        },
        index=grid.index,
    )
    df.index.name = "origin"
    return df
