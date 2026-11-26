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


# --- threshold execution ---------------------------------------------------------------------


def execution_thresholds(value: float, spec: BatterySpec) -> tuple[float, float]:
    """Price thresholds implied by a marginal value of stored energy ``value`` ($/MWh stored).

    Discharging one MWh to the grid earns ``p - wear`` and uses ``1 / eta`` MWh of storage;
    charging one MWh from the grid costs ``p`` and adds ``eta`` MWh. So: discharge if
    ``p > value / eta + wear``; charge if ``p < value * eta``; otherwise idle.
    """
    eta = spec.eta_one_way
    return value * eta, value / eta + spec.degradation_cost_per_mwh


def _execute(price: float, soc: float, charge_below: float, discharge_above: float, spec):
    """Full-power action for one interval at the realised price, within state-of-charge limits."""
    eta, dt = spec.eta_one_way, 5 / 60
    if price > discharge_above:
        d = min(spec.power_mw, max(soc - spec.soc_min_mwh, 0.0) * eta / dt)
        return 0.0, d, soc - d * dt / eta
    if price < charge_below:
        c = min(spec.power_mw, max(spec.soc_max_mwh - soc, 0.0) / (eta * dt))
        return c, 0.0, soc + eta * c * dt
    return 0.0, 0.0, soc


def threshold_dispatch(
    test_prices: pd.DataFrame,
    plan_prices: pd.DataFrame | None,
    curve: pd.Series,
    spec: BatterySpec,
    strategy: str,
    lookahead: int = 48,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    climatology: pd.Series | None = None,
    offer: str = "threshold",
) -> pd.DataFrame:
    """Re-plan every 5 minutes and execute by price threshold (or by an optimal offer curve).

    At each origin T (every interval):

    * With ``plan_prices`` (forecast-driven): plan the next ``lookahead`` intervals on the
      forecast planning prices, crediting leftover energy at the water value for the window's
      end, and take the marginal value of stored energy after the first interval (the LP dual).
    * Without (threshold-only benchmark): take the marginal value from the historical
      water-value ``curve`` for that interval. No forecasts.

    The interval is then executed by comparing the realised 5-minute price with the thresholds
    from :func:`execution_thresholds` (full power, within limits). This mirrors offering energy
    in price bands before the interval: under the price-taker assumption, a price-contingent offer
    is equivalent to acting on the price once it is set, so no forecast of the executed interval
    is needed. Simplification: the whole interval runs at full power, using the marginal value at
    the current state of charge (a real offer stack would use several bands).

    When the period's end first comes within ``lookahead`` intervals, the strategy makes one plan
    to the end that must finish at the initial state of charge (planning prices: the forecasts, or
    ``climatology`` for the threshold-only benchmark) and follows it: the same end rule as every
    other strategy, using only forecasts available at that moment.
    """
    from nemspike.battery.water_value import storage_values

    p = _full_days(add_nem_day(test_prices)).reset_index(drop=True)
    steps = p["interval_end"].diff().dropna()
    if not (steps == INTERVAL).all():
        raise ValueError("threshold dispatch needs an unbroken 5-minute price series")
    if plan_prices is None and climatology is None:
        raise ValueError("the threshold-only benchmark needs a climatology for the period's end")
    n = len(p)
    ends = p["interval_end"]
    actual = p["rrp"].to_numpy()
    curve_now = lookup_value(curve, ends)  # historical marginal value per interval
    slots = p["slot"].to_numpy()
    charge, discharge, soc_path = np.zeros(n), np.zeros(n), np.zeros(n)
    soc = spec.soc_initial_mwh

    if offer not in ("threshold", "curve"):
        raise ValueError("offer must be 'threshold' or 'curve'")
    if offer == "curve" and plan_prices is None:
        raise ValueError("an offer curve needs forecast plan prices")
    final_start = max(n - lookahead, 0)
    for i in range(final_start):
        window = lookahead
        if plan_prices is not None:
            origin = ends.iloc[i] - INTERVAL
            if origin not in plan_prices.index:
                raise KeyError(f"no forecasts for origin {origin}")
            path = plan_price_path(plan_prices.loc[origin], window, horizons)
            terminal = float(lookup_value(curve, ends.iloc[[i + window - 1]])[0])
            if offer == "curve":
                path = path.copy()
                path[0] = actual[i]  # the offer curve evaluated at the realised price
                plan = optimise(path, spec, soc_start=soc, terminal_value=terminal)
                charge[i], discharge[i] = plan.charge_mw[0], plan.discharge_mw[0]
                soc = float(plan.soc_mwh[0])
                soc_path[i] = soc
                continue
            value = float(storage_values(path, spec, soc_start=soc, terminal_value=terminal)[0])
        else:
            value = float(curve_now[i])
        below, above = execution_thresholds(value, spec)
        charge[i], discharge[i], soc = _execute(actual[i], soc, below, above, spec)
        soc_path[i] = soc

    # The period's end is in view: one plan to the end, finishing at the initial state of charge,
    # followed without re-planning (it needs only forecasts available at that moment).
    i, window = final_start, n - final_start
    if plan_prices is not None:
        origin = ends.iloc[i] - INTERVAL
        if origin not in plan_prices.index:
            raise KeyError(f"no forecasts for origin {origin}")
        path = plan_price_path(plan_prices.loc[origin], window, horizons)
    else:
        path = climatology.reindex(slots[i:]).to_numpy(dtype=float)
    plan = optimise(path, spec, soc_start=soc, soc_end=spec.soc_initial_mwh)
    charge[i:], discharge[i:], soc_path[i:] = plan.charge_mw, plan.discharge_mw, plan.soc_mwh

    return _record(p, Dispatch(charge, discharge, soc_path), strategy)
