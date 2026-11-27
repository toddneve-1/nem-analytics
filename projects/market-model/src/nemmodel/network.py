"""Build and solve a five-region PyPSA network for one year of hourly dispatch.

Representation (v0.1):

* one bus per NEM region, with operational demand as the load;
* interconnectors as controllable links with seasonal forward and reverse limits and no losses;
* thermal generators aggregated by power station, with capacity-weighted SRMC, seasonal ratings
  and (optionally) sampled forced outages;
* wind and solar aggregated by region, available up to their trace capacity factors;
* batteries aggregated by region, pumped hydro by station, conventional hydro by region as
  storage with a constant inflow sized to an annual capacity factor;
* a load-shedding generator in each region priced at VoLL, whose output is unserved energy;
* optionally, a daily cap on gas burnt by southern gas-powered generation in winter.

The year is solved in consecutive windows (a week by default) with perfect foresight inside each
window and storage returning to its starting level at the end of each window.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

REGIONS = ["QLD1", "NSW1", "VIC1", "SA1", "TAS1"]
THERMAL = ["coal", "ccgt", "ocgt", "recip", "liquid", "biomass"]
GAS_GROUPS = ["ccgt", "ocgt", "recip"]
SEASONAL = "seasonal_hydro"  # storage planned over the whole year (Tasmania, Snowy)
# linopy announces future changes to how it aligns coordinates ("legacy" vs "v1" semantics); the
# behaviour PyPSA relies on today is the legacy one, so these notices are silenced.
warnings.filterwarnings("ignore", message=".*Under v1.*")
warnings.filterwarnings("ignore", message=".*absent slots.*")
IO_API = "direct"  # pass the problem to HiGHS in memory (no LP file): faster, less memory
WHEEL_COST = 0.01  # $/MWh on interconnector flow, to stop flows that serve no purpose


@dataclass
class GasConstraint:
    """Daily cap on gas burnt by gas-powered generation in ``regions`` during ``months``: one
    figure for every day, or a Series of TJ indexed by day (midnight timestamps)."""

    regions: list[str]
    months: list[int]
    cap_tj_per_day: float | pd.Series


def unit_ratings(units: pd.DataFrame, index: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """Rated MW of each unit per hour (hours x unit_id) from the workbook's seasonal ratings.

    Summer months use the summer typical rating; winter months use the rating for that calendar
    year's winter. Units without a rating use their nameplate capacity.
    """
    summer = set(cfg["seasons"]["summer"])
    first_year = index[0].year
    cap = units["capacity_mw"].to_numpy(float)
    cols = {
        "summer": units["rating_summer_typical"],
        "winter_early": units["rating_winter_early"],
        "winter_late": units["rating_winter_late"],
    }
    rated = {k: v.fillna(units["capacity_mw"]).to_numpy(float) for k, v in cols.items()}
    out = np.empty((len(index), len(units)))
    for i, ts in enumerate(index):
        if ts.month in summer:
            out[i] = rated["summer"]
        elif ts.year == first_year:
            out[i] = rated["winter_early"]
        else:
            out[i] = rated["winter_late"]
    out = np.where(np.isfinite(out), out, cap)
    return pd.DataFrame(out, index=index, columns=units["unit_id"].to_numpy())


def monthly_inflow_mw(
    inflows: pd.DataFrame, scheme: str, reference_year: int, index: pd.DatetimeIndex
) -> pd.Series:
    """Hourly inflow (MW) from monthly inflow energy (GWh), spread evenly within each month.

    The weather reference year is a financial year (July to June), matched to the model's
    months by calendar month."""
    t = inflows[(inflows["scheme"] == scheme) & (inflows["reference_year"] == reference_year)]
    if t.empty:
        raise ValueError(f"no {scheme} inflows for reference year {reference_year}")
    gwh = t.set_index("month")["inflow"]
    month = index.strftime("%b")
    hours_in_month = pd.Series(1, index=index).groupby([index.year, index.month]).transform("sum")
    return pd.Series(gwh.reindex(month).to_numpy() * 1000, index=index) / hours_in_month


def build_network(
    units: pd.DataFrame,
    interconnectors: pd.DataFrame,
    demand: pd.DataFrame,
    vre_cf: pd.DataFrame,
    cfg: dict,
    available: pd.DataFrame | None = None,
    hydro_inflows: pd.DataFrame | None = None,
) -> pypsa.Network:
    """Network over ``demand.index`` (hourly). ``demand`` has one column per region (MW).
    ``available`` is the sampled available fraction per unit (hours x unit_id); None means no
    forced outages. ``hydro_inflows`` is the workbook's monthly inflow table; it is required if
    the fleet has Tasmanian or Snowy hydro."""
    idx = demand.index
    d = cfg["dispatch"]
    n = pypsa.Network()
    n.set_snapshots(idx)
    carriers = [
        "AC",
        "unserved",
        "interconnector",
        "battery",
        "pumped_hydro",
        "hydro",
        SEASONAL,
        "wind",
        "solar",
        *THERMAL,
    ]
    n.add("Carrier", carriers)
    n.add("Bus", REGIONS, carrier="AC")
    n.add("Load", REGIONS, suffix=" load", bus=REGIONS, p_set=demand[REGIONS])
    n.add(
        "Generator",
        REGIONS,
        suffix=" unserved",
        bus=REGIONS,
        carrier="unserved",
        p_nom=demand[REGIONS].max().to_numpy() * 1.1,
        marginal_cost=d["voll"],
    )

    mw = unit_ratings(units, idx, cfg)
    if available is not None:
        mw = mw * available.set_axis(idx).reindex(columns=mw.columns, fill_value=1.0)

    # Thermal: by power station (or by unit for committable groups in "exact_units" mode).
    uc = cfg.get("unit_commitment", {"enabled": False})
    uc_groups = set(uc["groups"]) if uc.get("enabled") else set()
    th = units[units["group"].isin(THERMAL)].copy()
    th["key"] = th["station"]
    if uc.get("mode") == "exact_units":
        th.loc[th["group"].isin(uc_groups), "key"] = th["unit_id"]
    for (region, key, group), g in th.groupby(["region", "key", "group"]):
        avail_mw = mw[g["unit_id"]].sum(axis=1)
        p_nom = float(avail_mw.max()) or float(g["capacity_mw"].sum())
        w = g["capacity_mw"]
        attrs = {}
        if group in uc_groups:
            # A breakdown or a zero seasonal rating overrides the minimum: the minimum is capped
            # at what the station can produce that hour (otherwise a station required to stay
            # on, for its minimum up time, could not, and the problem would have no solution).
            msl_pu = float(g["msl_mw"].sum()) / p_nom
            attrs = {
                "committable": True,
                "p_min_pu": np.minimum(msl_pu, avail_mw / p_nom),
                "min_up_time": int(uc["min_up_hours"][group]),
                "min_down_time": int(uc["min_down_hours"][group]),
                # Start the year free to stay on or switch off (not locked on).
                "up_time_before": int(uc["min_up_hours"][group]),
            }
        # Ramp-up limits only. A breakdown takes a unit offline at once, but a ramp-down limit
        # would stop the model doing so when the outage starts in the first hour of a window
        # (within a window the model sees it coming and ramps down early; across a window
        # boundary it cannot), making the window infeasible.
        if g["ramp_up_mw_min"].notna().all():
            up = float(g["ramp_up_mw_min"].sum()) * 60 / p_nom
            if up < 1:
                attrs["ramp_limit_up"] = up
        n.add(
            "Generator",
            f"{region} {key} {group}",
            bus=region,
            carrier=group,
            p_nom=p_nom,
            p_max_pu=avail_mw / p_nom,
            marginal_cost=float(np.average(g["srmc"], weights=w)),
            efficiency=float(3.6 / np.average(g["heat_rate"], weights=w))
            if g["heat_rate"].notna().all()
            else 1.0,
            **attrs,
        )

    # Wind and solar: by region and group.
    for col in vre_cf.columns:
        region, group = col.split("_")
        cap = units.loc[(units["region"] == region) & (units["group"] == group), "capacity_mw"]
        if cap.sum() > 0:
            n.add(
                "Generator",
                f"{region} {group}",
                bus=region,
                carrier=group,
                p_nom=float(cap.sum()),
                p_max_pu=vre_cf.loc[idx, col],
                marginal_cost=0.0,
            )

    # Storage.
    def add_storage(
        name: str,
        region: str,
        g: pd.DataFrame,
        carrier: str,
        eff: float,
        max_hours: float,
        inflow: float | pd.Series = 0.0,
        pump: bool = True,
    ) -> None:
        avail_mw = mw[g["unit_id"]].sum(axis=1)
        p_nom = float(avail_mw.max())
        if p_nom <= 0:
            return
        pu = avail_mw / p_nom
        n.add(
            "StorageUnit",
            name,
            bus=region,
            carrier=carrier,
            p_nom=p_nom,
            p_max_pu=pu,
            p_min_pu=-pu if pump else 0.0,
            max_hours=max_hours,
            efficiency_store=eff if pump else 1.0,
            efficiency_dispatch=eff if pump else 1.0,
            inflow=inflow,
            cyclic_state_of_charge=True,
            marginal_cost=0.001,
        )

    for region, g in units[units["group"] == "battery"].groupby("region"):
        hours = g["storage_mwh"].sum() / g["capacity_mw"].sum()
        add_storage(f"{region} battery", region, g, "battery", d["battery_efficiency"], hours)
    for (region, station), g in units[units["group"] == "pumped_hydro"].groupby(
        ["region", "station"]
    ):
        hours = g["storage_mwh"].sum() / g["capacity_mw"].sum()
        add_storage(
            f"{region} {station} pumped_hydro",
            region,
            g,
            "pumped_hydro",
            d["pumped_hydro_efficiency"],
            hours,
        )
    # Conventional hydro: Tasmania and Snowy are seasonal storage; the rest weekly.
    h = cfg["hydro"]
    levels: dict[str, dict] = {}
    hydro = units[units["group"] == "hydro"]
    is_tas = hydro["region"] == h["tasmania"]["region"]
    is_snowy = hydro["station"].isin(h["snowy"]["stations"])
    if (is_tas | is_snowy).any() and hydro_inflows is None:
        raise ValueError("hydro_inflows is required for Tasmanian and Snowy hydro")

    if is_tas.any():
        g = hydro[is_tas]
        energy = monthly_inflow_mw(hydro_inflows, "tasmania", cfg["trace_reference_year"], idx)
        p_nom = float(mw[g["unit_id"]].sum(axis=1).max())
        levels[f"{h['tasmania']['region']} tasmania hydro"] = h["tasmania"]
        add_storage(f"{h['tasmania']['region']} tasmania hydro", h["tasmania"]["region"], g,
                    SEASONAL, 1.0, h["tasmania"]["storage_gwh"] * 1000 / p_nom,
                    inflow=energy, pump=False)  # fmt: off
    if is_snowy.any():
        shape = monthly_inflow_mw(hydro_inflows, "snowy", cfg["trace_reference_year"], idx)
        shape = shape / shape.mean()  # representative only: keep the monthly pattern
        for region, g in hydro[is_snowy].groupby("region"):
            mean_mw = h["snowy"]["capacity_factor"] * float(g["capacity_mw"].sum())
            p_nom = float(mw[g["unit_id"]].sum(axis=1).max())
            levels[f"{region} snowy hydro"] = h["snowy"]
            add_storage(f"{region} snowy hydro", region, g, SEASONAL, 1.0,
                        h["snowy"]["storage_years"] * 8760 * mean_mw / p_nom,
                        inflow=shape * mean_mw, pump=False)  # fmt: off
    for region, g in hydro[~is_tas & ~is_snowy].groupby("region"):
        inflow = h["other"]["capacity_factor"] * float(g["capacity_mw"].sum())
        add_storage(f"{region} hydro", region, g, "hydro", 1.0, h["other"]["storage_hours"],
                    inflow=inflow, pump=False)  # fmt: off

    # Seasonal storage: start and end the year at a set level, never below a floor.
    for name, c in levels.items():
        if name in n.storage_units.index:
            energy = n.storage_units.at[name, "p_nom"] * n.storage_units.at[name, "max_hours"]
            n.storage_units.at[name, "cyclic_state_of_charge"] = False
            n.storage_units.at[name, "state_of_charge_initial"] = c["start_level"] * energy
    n.seasonal_levels = {k: (v["start_level"], v["min_level"]) for k, v in levels.items()
                         if k in n.storage_units.index}  # fmt: off

    # Interconnectors: seasonal limits.
    summer = idx.month.isin(cfg["seasons"]["summer"])
    for _, r in interconnectors.iterrows():
        fwd = pd.Series(np.where(summer, r["fwd_summer"], r["fwd_winter"]), index=idx)
        rev = pd.Series(np.where(summer, r["rev_summer"], r["rev_winter"]), index=idx)
        p_nom = float(max(fwd.max(), rev.max()))
        n.add(
            "Link",
            r["flow_path"],
            bus0=r["from"],
            bus1=r["to"],
            carrier="interconnector",
            p_nom=p_nom,
            p_max_pu=fwd / p_nom,
            p_min_pu=-rev / p_nom,
            marginal_cost=WHEEL_COST,
        )
    return n


def southern_gas_generators(n: pypsa.Network, regions: list[str]) -> pd.Index:
    g = n.generators
    return g.index[g["carrier"].isin(GAS_GROUPS) & g["bus"].isin(regions)]


def _gas_tj(n: pypsa.Network, gens: pd.Index) -> pd.Series:
    """Gas burnt (TJ per MWh of output) for each generator: heat rate = 3.6 / efficiency."""
    return 3.6 / n.generators.loc[gens, "efficiency"] / 1000.0


SIMPLEX_TIME_LIMIT = 120  # seconds; then the window is re-solved with interior point


def _optimise(n: pypsa.Network, sns: pd.Index, extra, linearised: bool = True) -> None:
    """Solve one window with HiGHS. The default simplex method occasionally stalls on a
    window (seen with some outage draws); after ``SIMPLEX_TIME_LIMIT`` seconds the window is
    re-solved with the interior-point method, and the window is recorded in
    ``n.solver_fallbacks``."""
    base = {"output_flag": False, "threads": 1}
    for options in ({**base, "time_limit": SIMPLEX_TIME_LIMIT},
                    {**base, "solver": "ipm", "run_crossover": "on"}):  # fmt: off
        status, cond = n.optimize(
            snapshots=sns,
            linearized_unit_commitment=linearised,
            io_api=IO_API,
            solver_name="highs",
            solver_options=options,
            extra_functionality=extra,
            include_objective_constant=False,
            progress=False,
        )
        if status == "ok":
            return
        if "time" not in str(cond).lower():
            break
        if not hasattr(n, "solver_fallbacks"):
            n.solver_fallbacks = []
        n.solver_fallbacks.append(str(sns[0]))
    raise RuntimeError(f"window starting {sns[0]}: {status} ({cond})")


def _planning_copy(n: pypsa.Network, step: int) -> pypsa.Network:
    """A copy of ``n`` for the whole-year water plan, in steps of ``step`` hours.

    Time series are averaged over each step and snapshot weightings set to ``step`` hours, so
    energy and storage balances still add up. On/off decisions, minimum levels and ramp limits
    are left out: they barely change how water is spread over a year but make the whole-year
    solve several times slower and larger.
    """
    m = n.copy()
    m.generators["committable"] = False
    m.generators["ramp_limit_up"] = np.nan
    m.generators["ramp_limit_down"] = np.nan
    m.generators_t.p_min_pu = pd.DataFrame(index=n.snapshots)
    if step == 1:
        return m
    if len(n.snapshots) % step:
        raise ValueError(f"{len(n.snapshots)} hours is not a whole number of {step}-hour steps")
    block = np.arange(len(n.snapshots)) // step
    coarse_index = n.snapshots[::step]
    saved = {}
    for c in m.components:
        for attr, df in c.dynamic.items():
            if df.empty or attr in {"p", "q", "state_of_charge", "status", "start_up",
                                    "shut_down", "p0", "p1", "marginal_price"}:  # fmt: off
                continue
            agg = df.groupby(block).last() if attr.endswith("_set") else df.groupby(block).mean()
            saved[(c.name, attr)] = agg.set_axis(coarse_index)
    m.set_snapshots(coarse_index)
    m.snapshot_weightings.loc[:, :] = step
    for (name, attr), df in saved.items():
        m.components[name].dynamic[attr] = df
    return m


def solve(n: pypsa.Network, cfg: dict, gas: GasConstraint | None = None) -> pypsa.Network:
    """Dispatch the year in two stages.

    1. **Plan:** one solve over the whole year sets the level of the seasonal hydro storage
       (Tasmania, Snowy) through the year: it starts and ends at the configured level and never
       falls below the configured floor.
    2. **Dispatch:** consecutive windows of ``cfg['dispatch']['window_hours']`` hours. Batteries,
       pumped hydro and small hydro return to their starting level in each window; seasonal
       hydro starts and ends each window at the level the plan set.

    Without seasonal storage, only stage 2 runs.
    """
    hours = cfg["dispatch"]["window_hours"]
    linearised = cfg.get("unit_commitment", {}).get("mode", "linearised") == "linearised"
    gens = southern_gas_generators(n, gas.regions) if gas else pd.Index([])
    tj = _gas_tj(n, gens) if gas else None

    def gas_cap(n: pypsa.Network, sns: pd.Index) -> None:
        sns = pd.DatetimeIndex(sns)
        days = sns.floor("D")
        winter = sns.month.isin(gas.months)
        if not winter.any() or len(gens) == 0:
            return
        p = n.model["Generator-p"]
        dim = next(d for d in p.dims if d != "snapshot")
        p = p.sel({dim: list(gens), "snapshot": sns[winter]})
        coef = xr.DataArray(tj.to_numpy(), coords={dim: list(gens)}, dims=dim)
        hours_per_step = n.snapshot_weightings["generators"].loc[sns[winter]]
        coef = coef * xr.DataArray(hours_per_step.to_numpy(), coords={"snapshot": sns[winter]},
                                   dims="snapshot")  # fmt: off
        day = xr.DataArray(
            days[winter], coords={"snapshot": sns[winter]}, dims="snapshot", name="day"
        )
        daily = (p * coef).sum(dim).groupby(day).sum()
        cap = gas.cap_tj_per_day
        if isinstance(cap, pd.Series):
            days_here = pd.DatetimeIndex(daily.indexes["day"])
            missing = days_here.difference(cap.index)
            if len(missing):
                raise ValueError(f"no gas allowance for {len(missing)} days, e.g. {missing[0]}")
            cap = xr.DataArray(cap.loc[days_here].to_numpy(), coords={"day": days_here},
                               dims="day")  # fmt: off
        n.model.add_constraints(daily <= cap, name="southern_gas_daily")

    seasonal = n.storage_units.index[n.storage_units["carrier"] == SEASONAL]
    energy = n.storage_units.loc[seasonal, "p_nom"] * n.storage_units.loc[seasonal, "max_hours"]
    floor = pd.Series({k: getattr(n, "seasonal_levels", {}).get(k, (0, 0))[1] for k in seasonal})

    def extra(n: pypsa.Network, sns: pd.Index) -> None:
        if gas:
            gas_cap(n, sns)
        if len(seasonal):
            soc = n.model["StorageUnit-state_of_charge"]
            dim = next(d for d in soc.dims if d != "snapshot")
            low = xr.DataArray((floor * energy).to_numpy(), coords={dim: list(seasonal)}, dims=dim)
            n.model.add_constraints(soc.sel({dim: list(seasonal)}) >= low, name="seasonal_floor")

    plan = None
    if len(seasonal):
        # Stage 1: the whole year, ending at the starting level.
        targets = pd.DataFrame(np.nan, index=n.snapshots, columns=seasonal)
        targets.iloc[-1] = n.storage_units.loc[seasonal, "state_of_charge_initial"]
        n.storage_units_t.state_of_charge_set = targets
        start_level = n.storage_units.loc[seasonal, "state_of_charge_initial"].copy()
        step = cfg["dispatch"].get("plan_step_hours", 1)
        if hours % step:
            raise ValueError(f"window_hours ({hours}) must be a multiple of plan_step_hours")
        m = _planning_copy(n, step)
        _optimise(m, m.snapshots, extra, linearised)
        # Plan levels at the end of each coarse step, placed on the hour that ends the step.
        coarse = m.storage_units_t.state_of_charge[seasonal]
        plan = pd.DataFrame(np.nan, index=n.snapshots, columns=seasonal)
        plan.iloc[step - 1 :: step] = coarse.to_numpy()
        del m

    for start in range(0, len(n.snapshots), hours):
        sns = n.snapshots[start : start + hours]
        if plan is not None:
            before = start_level if start == 0 else plan.iloc[start - 1]
            n.storage_units.loc[seasonal, "state_of_charge_initial"] = before
            if start > 0:
                n.storage_units_t.state_of_charge.loc[n.snapshots[start - 1], seasonal] = before
            targets = pd.DataFrame(np.nan, index=n.snapshots, columns=seasonal)
            targets.loc[sns[-1]] = plan.loc[sns[-1]]
            n.storage_units_t.state_of_charge_set = targets
        _optimise(n, sns, extra, linearised)  # stage 2

    if plan is not None:
        n.storage_units.loc[seasonal, "state_of_charge_initial"] = start_level
        n.storage_units_t.state_of_charge_set = pd.DataFrame(index=n.snapshots)
        n.seasonal_plan = plan  # kept for inspection and tests
    return n


def daily_southern_gas_tj(n: pypsa.Network, regions: list[str]) -> pd.Series:
    gens = southern_gas_generators(n, regions)
    burn = n.generators_t.p[gens].mul(_gas_tj(n, gens), axis=1).sum(axis=1)
    return burn.groupby(burn.index.floor("D")).sum()
