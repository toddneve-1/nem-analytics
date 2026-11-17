"""Battery dispatch strategies, evaluated at actual prices.

Two rule sets, applied identically to every strategy so that they differ only in information:

* **continuous** (default): the state of charge is carried across midnights; the battery starts
  and ends each evaluation period (a test quarter) at the initial level. This is how a real
  operator runs a battery and gives the true perfect-foresight ceiling.
* **daily**: every NEM day (288 intervals, 00:00-24:00 AEST) starts and ends at the initial level,
  so days are independent. Kept as a comparison.

Strategies:

* **Perfect foresight**: optimise on actual prices (the whole period, or each day). An upper bound
  no real strategy can reach.
* **Naive time-of-day**: optimise once on the *average* price profile by time of day from the
  training period (data before the test fold), then repeat that fixed schedule every day. Uses no
  forecasts and no information from the test period. Its schedule returns to the initial level
  every night, so it is valid under both rule sets.
* Forecast-driven rolling-horizon dispatch is added in Step 9.

Revenue is always settled at the actual RRP: ``sum(price * (discharge - charge) * 5/60)``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from nemspike.battery.lp import BatterySpec, Dispatch, optimise
from nemspike.timeutils import INTERVAL, to_nem_time

INTERVALS_PER_DAY = 288

Mode = Literal["continuous", "daily"]


def add_nem_day(prices: pd.DataFrame) -> pd.DataFrame:
    """Add ``nem_day`` (date of the interval start in NEM time) and ``slot`` (0..287)."""
    out = prices.sort_values("interval_end").copy()
    start = to_nem_time(out["interval_end"] - INTERVAL)
    out["nem_day"] = start.dt.date
    out["slot"] = start.dt.hour * 12 + start.dt.minute // 5
    return out


def _full_days(prices: pd.DataFrame) -> pd.DataFrame:
    counts = prices.groupby("nem_day")["rrp"].transform("count")
    return prices[
        (counts == INTERVALS_PER_DAY)
        & prices.groupby("nem_day")["rrp"].transform(lambda s: s.notna().all())
    ]


def _record(day_rows: pd.DataFrame, d: Dispatch, strategy: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "interval_end": day_rows["interval_end"].to_numpy(),
            "nem_day": day_rows["nem_day"].to_numpy(),
            "strategy": strategy,
            "rrp": day_rows["rrp"].to_numpy(),
            "charge_mw": d.charge_mw,
            "discharge_mw": d.discharge_mw,
            "soc_mwh": d.soc_mwh,
        }
    )


def perfect_foresight(
    prices: pd.DataFrame, spec: BatterySpec, mode: Mode = "continuous"
) -> pd.DataFrame:
    """Per-interval dispatch with full knowledge of actual prices.

    ``mode="continuous"``: one optimisation over the whole period, with the state of charge carried
    across midnights and fixed at the initial level only at the start and end of the period.
    ``mode="daily"``: each NEM day optimised separately, starting and ending at the initial level.
    """
    p = _full_days(add_nem_day(prices))
    if mode == "continuous":
        steps = p["interval_end"].diff().dropna()
        if not (steps == INTERVAL).all():
            raise ValueError("continuous optimisation needs an unbroken 5-minute price series")
        d = optimise(
            p["rrp"].to_numpy(), spec, soc_start=spec.soc_initial_mwh, soc_end=spec.soc_initial_mwh
        )
        return _record(p, d, "perfect_foresight")
    if mode != "daily":
        raise ValueError(f"unknown mode {mode!r}")
    out = []
    for _, day in p.groupby("nem_day", sort=True):
        d = optimise(
            day["rrp"].to_numpy(),
            spec,
            soc_start=spec.soc_initial_mwh,
            soc_end=spec.soc_initial_mwh,
        )
        out.append(_record(day, d, "perfect_foresight"))
    return pd.concat(out, ignore_index=True)


def time_of_day_schedule(train_prices: pd.DataFrame, spec: BatterySpec) -> Dispatch:
    """Optimal schedule for the average training-period price at each 5-minute slot of the day."""
    p = add_nem_day(train_prices)
    profile = p.groupby("slot")["rrp"].mean().reindex(range(INTERVALS_PER_DAY))
    if profile.isna().any():
        raise ValueError("training data does not cover every 5-minute slot of the day")
    return optimise(
        profile.to_numpy(), spec, soc_start=spec.soc_initial_mwh, soc_end=spec.soc_initial_mwh
    )


def naive_time_of_day(
    train_prices: pd.DataFrame, test_prices: pd.DataFrame, spec: BatterySpec
) -> pd.DataFrame:
    """Apply the training-period average-day schedule to every full test day."""
    schedule = time_of_day_schedule(train_prices, spec)
    p = _full_days(add_nem_day(test_prices))
    out = []
    for _, day in p.groupby("nem_day", sort=True):
        if not np.array_equal(day["slot"].to_numpy(), np.arange(INTERVALS_PER_DAY)):
            raise ValueError("day is not a complete, ordered set of 5-minute slots")
        out.append(_record(day, schedule, "naive_time_of_day"))
    return pd.concat(out, ignore_index=True)


def daily_results(dispatch: pd.DataFrame, spec: BatterySpec) -> pd.DataFrame:
    """Per strategy per day: market revenue, energy discharged, degradation cost and net revenue."""
    dt = 5 / 60
    d = dispatch.assign(
        market_revenue=dispatch["rrp"] * (dispatch["discharge_mw"] - dispatch["charge_mw"]) * dt,
        discharged_mwh=dispatch["discharge_mw"] * dt,
    )
    out = d.groupby(["strategy", "nem_day"], as_index=False)[
        ["market_revenue", "discharged_mwh"]
    ].sum()
    out["degradation_cost"] = out["discharged_mwh"] * spec.degradation_cost_per_mwh
    out["net_revenue"] = out["market_revenue"] - out["degradation_cost"]
    return out


def summarise_strategies(daily: pd.DataFrame, spec: BatterySpec) -> pd.DataFrame:
    """Totals per strategy; capture is net revenue relative to perfect foresight, same days."""
    usable = spec.soc_max_mwh - spec.soc_min_mwh
    g = daily.groupby("strategy")
    out = pd.DataFrame(
        {
            "days": g["nem_day"].nunique(),
            "market_revenue": g["market_revenue"].sum(),
            "degradation_cost": g["degradation_cost"].sum(),
            "net_revenue": g["net_revenue"].sum(),
            "net_per_day": g["net_revenue"].mean(),
            "share_of_days_negative": g["net_revenue"].apply(lambda s: float((s < 0).mean())),
            "cycles_per_day": g["discharged_mwh"].mean() / usable,
        }
    )
    out["net_per_mw_year"] = out["net_per_day"] * 365 / spec.power_mw
    if "perfect_foresight" in out.index:
        out["capture_vs_perfect"] = out["net_revenue"] / out.loc["perfect_foresight", "net_revenue"]
    return out
