"""Battery energy-arbitrage optimiser (linear program, optionally mixed-integer).

For a price path ``p_t`` over ``N`` 5-minute intervals (``dt = 1/12`` h), choose charge power
``c_t`` and discharge power ``d_t`` (MW, measured at the grid connection) and state of charge
``s_t`` (MWh, at the end of interval t) to

    maximise   sum_t p_t * (d_t - c_t) * dt                     (market revenue, $)

    subject to s_t = s_{t-1} + eta_c * c_t * dt - d_t * dt / eta_d   (energy balance)
               0 <= c_t, d_t <= P_max                             (power limits)
               S_min <= s_t <= S_max                              (state-of-charge limits)
               s_{N-1} = s_end                                    (terminal state of charge)

Round-trip efficiency ``eta`` is split equally between charging and discharging:
``eta_c = eta_d = sqrt(eta)``.

**Simultaneous charge and discharge.** With negative prices, the pure LP can profit by charging
and discharging at the same time: the round-trip losses turn the battery into a load that is paid
to consume. A real battery has one inverter and cannot do this, and forbidding it makes the
problem non-convex. ``optimise`` offers three methods:

* ``"lp"``: the plain linear program. Fast; an *upper bound* that may burn energy.
* ``"milp"``: exact mixed-integer program with a binary per negative-price interval. Used to
  validate the others; on days with hours of negative prices it can take minutes.
* ``"hybrid"`` (default): solve the LP. If it has no simultaneous operation at negative prices it
  is provably optimal for the real problem, and any overlap at prices >= 0 is removed exactly
  (see :func:`_remove_overlap`). Otherwise fall back to the dynamic programme in
  :mod:`nemspike.battery.dp`, which is exact on a fine state-of-charge grid and takes a few
  milliseconds. On 40 sampled real days the hybrid was within 0.02 % of the exact MILP revenue.

Assumptions: price taker (dispatch does not move the price), no degradation cost, no auxiliary
load, constant efficiency, no ramp limits, energy market only (no FCAS).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from nemspike.config import BatteryConfig

log = logging.getLogger(__name__)

DT_HOURS = 5 / 60
OVERLAP_TOL = 1e-6  # MW
MIP_TIME_LIMIT_S = 600.0

Method = Literal["hybrid", "lp", "milp"]


@dataclass(frozen=True)
class BatterySpec:
    power_mw: float
    energy_mwh: float
    round_trip_efficiency: float
    soc_min_mwh: float
    soc_max_mwh: float
    soc_initial_mwh: float

    @classmethod
    def from_config(cls, cfg: BatteryConfig) -> BatterySpec:
        e = cfg.energy_mwh
        return cls(
            power_mw=cfg.power_mw,
            energy_mwh=e,
            round_trip_efficiency=cfg.round_trip_efficiency,
            soc_min_mwh=cfg.soc_min_frac * e,
            soc_max_mwh=cfg.soc_max_frac * e,
            soc_initial_mwh=cfg.soc_initial_frac * e,
        )

    @property
    def eta_one_way(self) -> float:
        return float(np.sqrt(self.round_trip_efficiency))


@dataclass
class Dispatch:
    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray  # state of charge at the END of each interval

    @property
    def net_mw(self) -> np.ndarray:
        """Positive = exporting to the grid (discharging)."""
        return self.discharge_mw - self.charge_mw

    def revenue(self, prices: np.ndarray) -> float:
        """Revenue ($) of this dispatch at the given prices."""
        return float(np.sum(np.asarray(prices) * self.net_mw * DT_HOURS))

    def throughput_mwh(self) -> float:
        """Energy discharged to the grid (MWh)."""
        return float(np.sum(self.discharge_mw) * DT_HOURS)


def _solve(
    p: np.ndarray,
    spec: BatterySpec,
    s0: float,
    soc_end: float | None,
    binary_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One LP / MILP solve. Intervals in ``binary_idx`` may not charge and discharge together."""
    n = len(p)
    nb = len(binary_idx)
    eta = spec.eta_one_way
    dt = DT_HOURS

    # Variable layout: [c (n), d (n), s (n), u (nb)]
    nv = 3 * n + nb
    ic, id_, is_ = 0, n, 2 * n

    # Objective (milp minimises): -revenue = sum p*dt*c - p*dt*d
    cost = np.zeros(nv)
    cost[ic : ic + n] = p * dt
    cost[id_ : id_ + n] = -p * dt

    # Energy balance: s_t - s_{t-1} - eta*dt*c_t + dt/eta*d_t = (s0 if t == 0 else 0)
    eye = sparse.identity(n, format="csr")
    shift = sparse.diags([np.ones(n - 1)], [-1], shape=(n, n), format="csr")
    blocks = [-eta * dt * eye, (dt / eta) * eye, eye - shift]
    if nb:
        blocks.append(sparse.csr_matrix((n, nb)))
    a_bal = sparse.hstack(blocks, format="csr")
    rhs = np.zeros(n)
    rhs[0] = s0
    constraints = [LinearConstraint(a_bal, rhs, rhs)]

    if nb:
        # For interval k = binary_idx[j]: c_k <= P * u_j and d_k <= P * (1 - u_j).
        pmax = spec.power_mw
        pick = sparse.csr_matrix((np.ones(nb), (np.arange(nb), binary_idx)), shape=(nb, n))
        zero_n = sparse.csr_matrix((nb, n))
        u = sparse.identity(nb, format="csr")
        a_c = sparse.hstack([pick, zero_n, zero_n, -pmax * u], format="csr")
        a_d = sparse.hstack([zero_n, pick, zero_n, pmax * u], format="csr")
        constraints.append(LinearConstraint(a_c, -np.inf, 0.0))
        constraints.append(LinearConstraint(a_d, -np.inf, pmax))

    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    ub[ic : ic + n] = spec.power_mw
    ub[id_ : id_ + n] = spec.power_mw
    lb[is_ : is_ + n] = spec.soc_min_mwh
    ub[is_ : is_ + n] = spec.soc_max_mwh
    if soc_end is not None:
        lb[is_ + n - 1] = ub[is_ + n - 1] = soc_end
    integrality = np.zeros(nv)
    if nb:
        ub[3 * n :] = 1.0
        integrality[3 * n :] = 1

    res = milp(
        cost,
        constraints=constraints,
        bounds=Bounds(lb, ub),
        integrality=integrality,
        options={"disp": False, "mip_rel_gap": 1e-6, "time_limit": MIP_TIME_LIMIT_S},
    )
    if res.x is None:
        raise RuntimeError(f"battery optimisation failed: {res.message}")
    if not res.success:
        # Time limit reached with a feasible solution: use it, but say so.
        log.warning("battery MILP stopped early (%s); using best solution found", res.message)
    x = res.x
    return (
        np.clip(x[ic : ic + n], 0, None),
        np.clip(x[id_ : id_ + n], 0, None),
        x[is_ : is_ + n],
    )


def optimise(
    prices: np.ndarray,
    spec: BatterySpec,
    soc_start: float | None = None,
    soc_end: float | None = None,
    method: Method = "hybrid",
) -> Dispatch:
    """Revenue-maximising dispatch for a known price path.

    ``soc_start`` defaults to the spec's initial state of charge. ``soc_end`` fixes the final state
    of charge (``None`` leaves it free within limits). See the module docstring for ``method``.
    """
    p = np.asarray(prices, dtype=float)
    if len(p) == 0:
        raise ValueError("empty price path")
    if np.isnan(p).any():
        raise ValueError("price path contains NaN")
    s0 = spec.soc_initial_mwh if soc_start is None else soc_start
    eta = spec.eta_one_way

    if method == "milp":
        charge, discharge, soc = _solve(p, spec, s0, soc_end, np.flatnonzero(p < 0))
        charge, discharge = _remove_overlap(charge, discharge, eta)
        return Dispatch(charge_mw=charge, discharge_mw=discharge, soc_mwh=soc)

    charge, discharge, soc = _solve(p, spec, s0, soc_end, np.array([], dtype=int))
    if method == "lp":
        return Dispatch(charge_mw=charge, discharge_mw=discharge, soc_mwh=soc)
    if method != "hybrid":
        raise ValueError(f"unknown method {method!r}")
    if has_negative_price_overlap(charge, discharge, p):
        from nemspike.battery.dp import optimise_dp

        return optimise_dp(p, spec, soc_start=s0, soc_end=soc_end)
    charge, discharge = _remove_overlap(charge, discharge, eta)
    return Dispatch(charge_mw=charge, discharge_mw=discharge, soc_mwh=soc)


def has_negative_price_overlap(
    charge: np.ndarray, discharge: np.ndarray, prices: np.ndarray
) -> bool:
    """True if any negative-price interval both charges and discharges."""
    return bool(np.any((charge > OVERLAP_TOL) & (discharge > OVERLAP_TOL) & (prices < 0)))


def _remove_overlap(
    charge: np.ndarray, discharge: np.ndarray, eta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Remove simultaneous charge/discharge without changing the state of charge.

    Reducing charge by x and discharge by eta^2 * x keeps ``eta * c - d / eta`` unchanged and
    changes revenue by ``p * (1 - eta^2) * x * dt``, which is >= 0 when p >= 0. It is only applied
    where no negative-price interval overlaps (or after the MILP has forbidden those), so it never
    loses revenue.
    """
    c, d = charge.copy(), discharge.copy()
    x = np.minimum(c, d / eta**2)
    c -= x
    d -= eta**2 * x
    return np.clip(c, 0, None), np.clip(d, 0, None)
