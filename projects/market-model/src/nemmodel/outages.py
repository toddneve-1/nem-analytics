"""Monte Carlo forced outages, unit by unit, at hourly resolution.

Each unit follows two independent two-state Markov chains, one for full outages and one for
partial outages, parameterised by the workbook's outage rates and mean times to repair:

* repair probability per hour ``r = 1 / MTTR``
* failure probability per hour ``f = FOR * r / (1 - FOR)``, which makes the long-run share of
  hours on outage equal to the forced outage rate ``FOR``

A full outage removes the unit; a partial outage removes ``partial_derate`` of its capacity. Each
chain starts in its stationary distribution. Wind and solar carry no outages here: their traces
already describe availability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _chain(rng: np.random.Generator, rate: np.ndarray, mttr: np.ndarray, hours: int) -> np.ndarray:
    """Boolean (hours x units) outage states for one two-state chain per unit."""
    rate = np.nan_to_num(rate, nan=0.0)
    mttr = np.where(np.isfinite(mttr) & (mttr > 0), mttr, 1.0)
    active = rate > 0
    rate = np.minimum(rate, 0.999)
    repair = 1.0 / np.maximum(mttr, 1.0)
    fail = np.where(active, rate * repair / (1.0 - rate), 0.0)
    out = np.zeros((hours, rate.size), dtype=bool)
    state = active & (rng.random(rate.size) < rate)
    u = rng.random((hours, rate.size))
    for t in range(hours):
        state = np.where(state, u[t] >= repair, u[t] < fail)
        out[t] = state
    return out


def sample_available_fraction(
    units: pd.DataFrame, hours: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Available fraction of each unit's capacity per hour (hours x unit_id), between 0 and 1."""
    full = _chain(rng, units["for_full"].to_numpy(float), units["mttr_full"].to_numpy(float), hours)
    partial = _chain(
        rng, units["for_partial"].to_numpy(float), units["mttr_partial"].to_numpy(float), hours
    )
    derate = np.nan_to_num(units["partial_derate"].to_numpy(float), nan=0.0)
    avail = np.where(full, 0.0, np.where(partial, 1.0 - derate, 1.0))
    return pd.DataFrame(avail, columns=units["unit_id"].to_numpy())
