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

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from nemspike.battery.lp import DT_HOURS, OVERLAP_TOL, BatterySpec, optimise


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
