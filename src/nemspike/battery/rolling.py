"""Forecast-driven rolling-horizon battery dispatch.

Every ``resolve`` intervals (30 minutes by default) at forecast origin T:

1. Build a planning price path for the next ``lookahead`` intervals (4 hours) from the forecasts
   made at T. Forecasts exist on a 30-minute grid of horizons (30, 60, ..., 240 minutes) for the
   interval ending at T + h; each is applied to the six 5-minute intervals of its block
   (T + h - 30, T + h].
2. Optimise that window (same battery, efficiency, wear cost and exclusivity rules as every other
   strategy). Energy still stored at the window's end is credited at the **water value** for the
   time the window ends (weekday/weekend curve learned from history), so the plan neither treats
   leftover energy as worthless nor hoards it.
3. Execute only the first ``resolve`` intervals of the plan, settled later at actual prices; carry
   the state of charge forward; repeat with fresh forecasts.

Rules shared with perfect foresight: the battery starts the evaluation period at the initial state
of charge, carries it continuously, and must finish the period at the initial level. Once the
period's end falls inside the window, the plan is required to end there at the initial level.

No look-ahead: the decision at T uses only the forecasts made at T, the water-value curve (from
data before the evaluation period) and the current state of charge. Actual prices are used only to
settle revenue afterwards. A test checks that changing later actual prices leaves earlier decisions
unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nemspike.battery.lp import BatterySpec, Dispatch, optimise
from nemspike.battery.strategies import _full_days, _record, add_nem_day
from nemspike.battery.water_value import lookup_value
from nemspike.timeutils import INTERVAL

DEFAULT_HORIZONS = (30, 60, 90, 120, 150, 180, 210, 240)


def plan_price_path(row: pd.Series, n_intervals: int, horizons: tuple[int, ...]) -> np.ndarray:
    """Planning price for each of the next ``n_intervals`` 5-minute intervals from one origin's
    forecasts (``row`` is indexed by horizon in minutes).

    Horizons must form an even grid (step, 2*step, ...). Each forecast, made for the single
    interval ending at origin + h, is applied to the whole block (h - step, h]. With the project's
    30-minute grid this smooths 5-minute detail: a forecast cannot place a spike in a specific
    5-minute interval within its block.
    """
    step = horizons[0]
    if list(horizons) != [step * (i + 1) for i in range(len(horizons))] or step % 5:
        raise ValueError("horizons must be an even grid of multiples of 5 minutes")
    minutes_ahead = 5 * (np.arange(n_intervals) + 1)
    block = step * np.ceil(minutes_ahead / step).astype(int)
    if block.max() > max(horizons):
        raise ValueError("planning window extends beyond the longest forecast horizon")
    path = row.reindex(block).to_numpy(dtype=float)
    if np.isnan(path).any():
        raise ValueError(f"missing forecasts for horizons {sorted(set(block[np.isnan(path)]))}")
    return path


def rolling_dispatch(
    test_prices: pd.DataFrame,
    plan_prices: pd.DataFrame,
    curve: pd.Series,
    spec: BatterySpec,
    strategy: str,
    resolve: int = 6,
    lookahead: int = 48,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Dispatch over a contiguous evaluation period.

    ``plan_prices`` is indexed by forecast origin (UTC) with one column per horizon (minutes),
    holding the planning price (for example the expected price) for the interval ending at
    origin + horizon. ``curve`` is the water-value curve by (weekend, slot).
    """
    p = _full_days(add_nem_day(test_prices)).reset_index(drop=True)
    steps = p["interval_end"].diff().dropna()
    if not (steps == INTERVAL).all():
        raise ValueError("rolling dispatch needs an unbroken 5-minute price series")
    n = len(p)
    ends = p["interval_end"]
    charge, discharge, soc_path = np.zeros(n), np.zeros(n), np.zeros(n)
    soc = spec.soc_initial_mwh

    for i in range(0, n, resolve):
        origin = ends.iloc[i] - INTERVAL
        if origin not in plan_prices.index:
            raise KeyError(f"no forecasts for origin {origin}")
        window = min(lookahead, n - i)
        prices = plan_prices.loc[origin]
        path = plan_price_path(prices, window, horizons)
        if i + window == n:  # the period's end is in view: finish at the initial level
            soc_end, terminal = spec.soc_initial_mwh, 0.0
        else:
            soc_end = None
            terminal = float(lookup_value(curve, ends.iloc[[i + window - 1]])[0])
        plan = optimise(path, spec, soc_start=soc, soc_end=soc_end, terminal_value=terminal)
        k = min(resolve, window)
        charge[i : i + k] = plan.charge_mw[:k]
        discharge[i : i + k] = plan.discharge_mw[:k]
        soc_path[i : i + k] = plan.soc_mwh[:k]
        soc = float(plan.soc_mwh[k - 1])

    return _record(p, Dispatch(charge, discharge, soc_path), strategy)
