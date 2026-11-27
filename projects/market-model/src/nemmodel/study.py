"""The 2 x 2 scenario study: coal retirement timing x winter gas availability.

For each scenario, each demand POE level and each outage sample, the year is dispatched and the
unserved energy recorded. Expected unserved energy (EUE) for a region is the POE-weighted mean
over samples, as a percentage of the region's 50 % POE annual operational demand, and is compared
with the reliability standard (0.002 %) and the Interim Reliability Measure (0.0006 %).

Outage samples use common random numbers: sample ``s`` draws the same outages for every unit in
every scenario (retired units are simply dropped), so differences between scenarios are not
sampling noise.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nemmodel.inputs import ModelYear
from nemmodel.network import REGIONS, GasConstraint, build_network, daily_southern_gas_tj, solve
from nemmodel.outages import sample_available_fraction
from nemmodel.traces import hourly_demand, regional_vre_capacity_factors

COAL_CASES = ["workbook", "early"]
GAS_CASES = ["unconstrained", "tight"]


@dataclass(frozen=True)
class Case:
    coal: str
    gas: str

    @property
    def name(self) -> str:
        return f"coal-{self.coal}_gas-{self.gas}"


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def fleet(units: pd.DataFrame, coal: str, cfg: dict) -> pd.DataFrame:
    """Units in service for a coal-timing case."""
    if coal == "workbook":
        return units
    by = cfg["coal_timing"]["early_closure_by"]
    early = (units["group"] == "coal") & (units["closure_year"] <= by)
    return units[~early]


@dataclass
class Data:
    units: pd.DataFrame
    interconnectors: pd.DataFrame
    demand: pd.DataFrame
    vre_cf: pd.DataFrame
    hydro_inflows: pd.DataFrame | None = None


def load_data(root: Path, cfg: dict, trace_dir: Path) -> Data:
    inputs = root / cfg["inputs"]
    units = pd.read_csv(inputs / "units.csv")
    ics = pd.read_csv(inputs / "interconnectors.csv")
    year = ModelYear(cfg["financial_year_end"])
    demand = hourly_demand(trace_dir, year, cfg["scenario"])
    cf, _ = regional_vre_capacity_factors(trace_dir, units, year)
    inflows = pd.read_csv(inputs / "hydro_inflows.csv")
    return Data(units, ics, demand, cf, inflows)


def demand_for(data: Data, poe: str) -> pd.DataFrame:
    cols = [f"{r}_{poe}" for r in REGIONS]
    return data.demand[cols].set_axis(REGIONS, axis=1)


def run_one(
    data: Data,
    cfg: dict,
    case: Case,
    poe: str,
    sample: int | None,
    gas_cap: float | pd.Series | None,
) -> dict:
    """Dispatch one year. ``sample`` None means no forced outages."""
    units = fleet(data.units, case.coal, cfg)
    dem = demand_for(data, poe)
    avail = None
    if sample is not None:
        rng = np.random.default_rng([cfg["reliability"]["seed"], sample])
        avail = sample_available_fraction(data.units, len(dem), rng)  # all units: common draws
    n = build_network(units, data.interconnectors, dem, data.vre_cf, cfg, avail,
                      data.hydro_inflows)  # fmt: off
    gas = None
    if case.gas == "tight":
        g = cfg["gas"]
        gas = GasConstraint(g["southern_regions"], g["constraint_months"], gas_cap)
    solve(n, cfg, gas)

    gen = n.generators_t.p.T.groupby(n.generators["carrier"]).sum().T.sum()
    sto = n.storage_units_t.p.clip(lower=0).T.groupby(n.storage_units["carrier"]).sum().T.sum()
    unserved = n.generators_t.p[[f"{r} unserved" for r in REGIONS]].set_axis(REGIONS, axis=1)
    gas_daily = daily_southern_gas_tj(n, cfg["gas"]["southern_regions"])
    winter = gas_daily.index.month.isin(cfg["gas"]["constraint_months"])
    return {
        "case": case.name,
        "coal": case.coal,
        "gas": case.gas,
        "poe": poe,
        "sample": sample,
        **{f"unserved_mwh_{r}": float(unserved[r].sum()) for r in REGIONS},
        **{f"unserved_hours_{r}": int((unserved[r] > 1e-3).sum()) for r in REGIONS},
        **{f"gen_twh_{k}": float(v) / 1e6 for k, v in pd.concat([gen, sto]).items()},
        "gas_max_winter_tj": float(gas_daily[winter].max()),
        "gas_total_pj": float(gas_daily.sum()) / 1000,
        "mean_price": {r: float(v) for r, v in n.buses_t.marginal_price.mean().items()},
    }


def _task(args: tuple) -> dict:
    logging.disable(logging.WARNING)
    data, cfg, case, poe, sample, cap = args
    return run_one(data, cfg, case, poe, sample, cap)


def eue_table(runs: pd.DataFrame, data: Data, cfg: dict) -> pd.DataFrame:
    """EUE (MWh and % of 50 % POE annual energy) by case and region."""
    w = cfg["reliability"]["poe_weights"]
    energy = demand_for(data, "POE50").sum()
    rows = []
    for case, g in runs.groupby("case"):
        for r in REGIONS:
            by_poe = g.groupby("poe")[f"unserved_mwh_{r}"].mean()
            eue = sum(w[p] * by_poe.get(p, np.nan) for p in w)
            rows.append(
                {
                    "case": case,
                    "region": r,
                    "eue_mwh": eue,
                    "eue_pct": 100 * eue / energy[r],
                    "draws_with_unserved": int(
                        (g.groupby("sample")[f"unserved_mwh_{r}"].max() > 1).sum()
                    ),
                    "draws": g["sample"].nunique(),
                }
            )
    t = pd.DataFrame(rows)
    t["meets_standard"] = t["eue_pct"] <= cfg["reliability"]["standard_pct"]
    t["meets_interim_measure"] = t["eue_pct"] <= cfg["reliability"]["interim_measure_pct"]
    return t


def tight_gas_cap(root: Path, cfg: dict, data: Data, out_dir: Path) -> float | pd.Series:
    """Daily gas allowance for southern gas-powered generation in the tight-gas case.

    ``illustrative``: a fixed fraction of the most the unconstrained model burns on a winter day
    (workbook fleet, 10 % POE, no outages). ``gsoo``: the southern gas balance in
    ``config/gas_balance.yaml``, which needs GSOO values. Days on which other gas demand alone
    exceeds supply leave power stations nothing; they are counted, and the gas shortfall to
    other users is reported but not modelled.
    """
    g = cfg["gas"]
    if g["tight_case"] == "gsoo":
        from nemmodel.gas import daily_allowance, load_balance

        balance = load_balance(root / "config" / "gas_balance.yaml")
        infra = pd.read_csv(root / cfg["inputs"] / "gas_infrastructure.csv")
        south = demand_for(data, "POE50")[g["southern_regions"]].sum(axis=1)
        raw = daily_allowance(balance, infra, south, g["constraint_months"])
        record = {"method": "gsoo", "source": balance["source"],
                  "allowance_tj_per_day": raw.round(1).astype(str).to_dict(),
                  "days_other_demand_exceeds_supply": int((raw < 0).sum()),
                  "other_users_shortfall_tj": float(-raw[raw < 0].sum())}  # fmt: off
        cap = raw.clip(lower=0)
    elif g["tight_case"] == "illustrative":
        # In a separate process, so the main process does not keep a solver's memory.
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
            args = (data, cfg, Case("workbook", "unconstrained"), "POE10", None, None)
            ref = ex.submit(_task, args).result()
        cap = g["tight_fraction"] * ref["gas_max_winter_tj"]
        record = {"method": "illustrative",
                  "reference_max_winter_day_tj": ref["gas_max_winter_tj"],
                  "tight_fraction": g["tight_fraction"], "tight_cap_tj_per_day": cap}  # fmt: off
    else:
        raise ValueError(f"unknown gas.tight_case: {g['tight_case']}")
    (out_dir / "gas_calibration.json").write_text(json.dumps(record, indent=2, default=str))
    return cap


def run_study(
    root: Path, trace_dir: Path, out_dir: Path, samples: int | None = None, workers: int = 3
) -> pd.DataFrame:
    cfg = load_config(root / "config" / "assumptions.yaml")
    data = load_data(root, cfg, trace_dir)
    samples = samples or cfg["reliability"]["samples"]
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = tight_gas_cap(root, cfg, data, out_dir)

    tasks = [
        (data, cfg, Case(c, g), poe, s, cap)
        for c in COAL_CASES
        for g in GAS_CASES
        for poe in cfg["reliability"]["poe_weights"]
        for s in range(samples)
    ]
    # "spawn", not fork: the parent has already run HiGHS, and forked children can deadlock.
    # Each worker peaks at about 3.3 GB during the whole-year hydro plan, so three workers fit
    # in 15 GB; workers are replaced after a few runs so memory does not creep up.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=4
    ) as ex:
        results = []
        for i, r in enumerate(ex.map(_task, tasks), 1):
            results.append(r)
            print(f"[{i}/{len(tasks)}] {r['case']} {r['poe']} sample {r['sample']}", flush=True)
    runs = pd.DataFrame(results)
    runs.drop(columns="mean_price").to_csv(out_dir / "runs.csv", index=False)
    eue = eue_table(runs, data, cfg)
    eue.to_csv(out_dir / "eue.csv", index=False)
    return eue
