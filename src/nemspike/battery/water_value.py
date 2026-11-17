"""Marginal value of stored energy ("water value") from the perfect-foresight optimisation.

The shadow price (dual) of the energy-balance constraint at interval t is the change in optimal
revenue from one extra MWh in storage at t, in $/MWh. Averaged over history by time of day, it
gives an empirical curve for valuing energy left in the battery at the end of a planning window.

Duals need a linear program. Forbidding simultaneous charge and discharge needs binaries, but once
the charge-or-discharge choice in each affected interval is fixed, the problem is an LP again and
its duals are exact for that choice. So:

1. Solve the plain LP. If it never charges and discharges together at a negative price, it is the
   true optimum and its duals are used directly.
2. Otherwise get the valid dispatch from the exact method (hybrid: dynamic programme fallback),
   forbid the unused direction in each overlapping interval, re-solve the LP, and repeat until no
   negative-price interval overlaps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import holidays
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from nemspike.battery.lp import DT_HOURS, OVERLAP_TOL, BatterySpec, optimise
from nemspike.timeutils import INTERVAL, to_nem_time


def _lp_with_duals(
    p: np.ndarray,
    spec: BatterySpec,
    s0: float,
    soc_end: float | None,
    charge_cap: np.ndarray,
    discharge_cap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Solve the LP with per-interval power caps. Returns charge, discharge, duals, revenue."""
    n, eta, dt, k = len(p), spec.eta_one_way, DT_HOURS, spec.degradation_cost_per_mwh
    cost = np.concatenate([p * dt, (-p + k) * dt, np.zeros(n)])
    eye = sparse.identity(n, format="csr")
    shift = sparse.diags([np.ones(n - 1)], [-1], shape=(n, n), format="csr")
    a_eq = sparse.hstack([-eta * dt * eye, (dt / eta) * eye, eye - shift], format="csr")
    b_eq = np.zeros(n)
    b_eq[0] = s0
    lb = np.concatenate([np.zeros(2 * n), np.full(n, spec.soc_min_mwh)])
    ub = np.concatenate([charge_cap, discharge_cap, np.full(n, spec.soc_max_mwh)])
    if soc_end is not None:
        lb[-1] = ub[-1] = soc_end
    res = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=list(zip(lb, ub, strict=True)), method="highs")
    if not res.success:
        raise RuntimeError(f"water-value LP failed: {res.message}")
    # The objective is -revenue, so the value of one more MWh of storage is minus the marginal.
    return res.x[:n], res.x[n : 2 * n], -res.eqlin.marginals, -float(res.fun)


def storage_values(
    prices: np.ndarray,
    spec: BatterySpec,
    soc_start: float | None = None,
    soc_end: float | None = None,
    max_rounds: int = 10,
) -> np.ndarray:
    """Marginal value ($/MWh) of one extra MWh of storage at each interval (see module notes)."""
    p = np.asarray(prices, dtype=float)
    n = len(p)
    s0 = spec.soc_initial_mwh if soc_start is None else soc_start
    charge_cap = np.full(n, spec.power_mw)
    discharge_cap = np.full(n, spec.power_mw)
    exact = None
    for _ in range(max_rounds):
        c, d, duals, _ = _lp_with_duals(p, spec, s0, soc_end, charge_cap, discharge_cap)
        overlap = (c > OVERLAP_TOL) & (d > OVERLAP_TOL) & (p < 0)
        if not overlap.any():
            return duals
        if exact is None:
            exact = optimise(p, spec, soc_start=s0, soc_end=soc_end, method="hybrid")
        # Keep the direction the exact solution uses (charging at a negative price if idle).
        uses_discharge = exact.discharge_mw > OVERLAP_TOL
        charge_cap[overlap & uses_discharge] = 0.0
        discharge_cap[overlap & ~uses_discharge] = 0.0
    raise RuntimeError("could not remove simultaneous charge/discharge for the water value")


# --- empirical water-value curve -------------------------------------------------------------


def day_type(interval_end: pd.Series) -> np.ndarray:
    """1 for weekends and Victorian public holidays, else 0 (by interval start in NEM time).

    Grouping chosen from data: weekday/weekend beat both a single curve and day-of-week curves
    out of sample (design notes 9.1).
    """
    start = to_nem_time(interval_end - INTERVAL)
    vic = holidays.Australia(subdiv="VIC", years=sorted(set(start.dt.year)))
    is_holiday = np.array([d in vic for d in start.dt.date])
    return ((start.dt.day_of_week.to_numpy() >= 5) | is_holiday).astype(int)


def slot_of_day(interval_end: pd.Series) -> np.ndarray:
    """5-minute slot of the NEM day (0..287) by interval start."""
    start = to_nem_time(interval_end - INTERVAL)
    return (start.dt.hour * 12 + start.dt.minute // 5).to_numpy()


def quarterly_storage_values(prices: pd.DataFrame, spec: BatterySpec) -> pd.DataFrame:
    """Storage values for every interval, optimising each NEM quarter continuously (50 % at the
    quarter's start and end, as in the perfect-foresight benchmark)."""
    p = prices.sort_values("interval_end").reset_index(drop=True)
    start = to_nem_time(p["interval_end"] - INTERVAL).dt.tz_localize(None)
    quarter = start.dt.to_period("Q")
    parts = []
    for _, chunk in p.groupby(quarter.to_numpy(), sort=True):
        v = storage_values(chunk["rrp"].to_numpy(), spec, soc_end=spec.soc_initial_mwh)
        parts.append(pd.Series(v, index=chunk.index))
    return pd.DataFrame(
        {
            "interval_end": p["interval_end"],
            "value": pd.concat(parts).sort_index().to_numpy(),
            "slot": slot_of_day(p["interval_end"]),
            "weekend": day_type(p["interval_end"]),
        }
    )


def storage_values_key(prices: pd.DataFrame, spec: BatterySpec) -> str:
    """Short fingerprint of everything :func:`quarterly_storage_values` depends on.

    Used to name the cache file, so cached values are never reused for a different battery
    (efficiency, limits, power, wear cost) or a different price history (period or revisions).
    """
    ends = prices["interval_end"]
    payload = {
        "spec": asdict(spec),
        "first": str(ends.min()),
        "last": str(ends.max()),
        "n": len(prices),
        "rrp": int(pd.util.hash_pandas_object(prices["rrp"], index=False).sum()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def water_value_curve(values: pd.DataFrame, before: pd.Timestamp) -> pd.Series:
    """Mean storage value by (weekend, slot) using only intervals ending at or before ``before``."""
    hist = values[values["interval_end"] <= before]
    curve = hist.groupby(["weekend", "slot"])["value"].mean()
    if len(curve) < 2 * 288:
        raise ValueError("history does not cover every slot for both weekdays and weekends")
    return curve


def lookup_value(curve: pd.Series, interval_end: pd.Series) -> np.ndarray:
    """Water value ($/MWh) at the given interval ends."""
    keys = pd.MultiIndex.from_arrays([day_type(interval_end), slot_of_day(interval_end)])
    return curve.reindex(keys).to_numpy()
