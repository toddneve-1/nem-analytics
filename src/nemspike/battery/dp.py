"""Exact-on-a-grid dynamic programme for battery arbitrage with no simultaneous charge/discharge.

Why this exists: forbidding simultaneous charging and discharging makes the problem non-convex.
A mixed-integer program solves it exactly, but on days with hours of negative prices (up to 200
of 288 intervals) the optimal plan alternates charge and discharge in intricate patterns and the
MILP can take minutes per day. A dynamic programme has a fixed, small cost regardless of prices.

Method: discretise state of charge into steps of ``g`` MWh. In each interval the battery moves by
``k`` steps: ``k > 0`` charges (buys ``k * g / eta`` MWh from the grid), ``k < 0`` discharges
(sells ``-k * g * eta`` MWh), ``k = 0`` idles. Charging and discharging in the same interval is
impossible by construction. Backward induction gives the optimal policy for every state.

The grid is anchored on the starting state of charge, so the start, and an end equal to the start,
are exact; any other target end state is snapped to the nearest grid point (within g / 2).

Grid choice: ``g = eta * P * dt / m``, so full charging power is exactly ``m`` steps. Full
discharging power is ``P * dt / eta = m / eta^2`` steps, which is an integer when ``m / RTE`` is
(for RTE = 0.9 and m = 18: 20 steps). Otherwise the discharge limit is rounded down slightly.
"""

from __future__ import annotations

import numpy as np

from nemspike.battery.lp import DT_HOURS, BatterySpec, Dispatch

DEFAULT_STEPS_PER_FULL_CHARGE = 18


def optimise_dp(
    prices: np.ndarray,
    spec: BatterySpec,
    soc_start: float | None = None,
    soc_end: float | None = None,
    m: int = DEFAULT_STEPS_PER_FULL_CHARGE,
) -> Dispatch:
    """Optimal exclusive dispatch on a state-of-charge grid (see module docstring)."""
    p = np.asarray(prices, dtype=float)
    n = len(p)
    eta = spec.eta_one_way
    dt = DT_HOURS
    g = eta * spec.power_mw * dt / m
    up = m
    down = int(np.floor(spec.power_mw * dt / eta / g + 1e-9))
    offsets = np.arange(-down, up + 1)

    # Anchor the grid on the starting state of charge so it is represented exactly; the usual
    # terminal condition (end where you started) is then exact too.
    s0 = spec.soc_initial_mwh if soc_start is None else soc_start
    k_lo = int(np.ceil((spec.soc_min_mwh - s0) / g - 1e-9))
    k_hi = int(np.floor((spec.soc_max_mwh - s0) / g + 1e-9))
    n_states = k_hi - k_lo + 1
    grid = s0 + np.arange(k_lo, k_hi + 1) * g
    i0 = -k_lo

    def to_index(soc: float) -> int:
        return int(np.argmin(np.abs(grid - soc)))

    value = np.zeros(n_states)
    if soc_end is not None:
        value = np.full(n_states, -np.inf)
        value[to_index(soc_end)] = 0.0

    # Revenue of moving k steps at price 1 $/MWh: charge pays k*g/eta, discharge earns -k*g*eta.
    unit_rev = np.where(offsets > 0, -offsets * g / eta, -offsets * g * eta)
    states = np.arange(n_states)
    nxt = states[:, None] + offsets[None, :]
    valid = (nxt >= 0) & (nxt < n_states)
    nxt_clipped = np.clip(nxt, 0, n_states - 1)
    policy = np.zeros((n, n_states), dtype=np.int16)

    for t in range(n - 1, -1, -1):
        q = p[t] * unit_rev[None, :] + value[nxt_clipped]
        q[~valid] = -np.inf
        best = np.argmax(q, axis=1)
        policy[t] = offsets[best]
        value = q[states, best]

    if not np.isfinite(value[i0]):
        raise RuntimeError("no feasible dispatch reaches the required final state of charge")

    moves = np.empty(n, dtype=int)
    idx = i0
    for t in range(n):
        moves[t] = policy[t, idx]
        idx += moves[t]
    charge = np.where(moves > 0, moves * g / (eta * dt), 0.0)
    discharge = np.where(moves < 0, -moves * g * eta / dt, 0.0)
    soc = grid[i0 + np.cumsum(moves)]
    return Dispatch(charge_mw=charge, discharge_mw=discharge, soc_mwh=soc)
