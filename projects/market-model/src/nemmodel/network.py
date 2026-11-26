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
warnings.filterwarnings("ignore", message=".*absent slots.*")
WHEEL_COST = 0.01  # $/MWh on interconnector flow, to stop flows that serve no purpose


@dataclass
class GasConstraint:
    regions: list[str]
    months: list[int]
    cap_tj_per_day: float


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


def build_network(
    units: pd.DataFrame,
    interconnectors: pd.DataFrame,
    demand: pd.DataFrame,
    vre_cf: pd.DataFrame,
    cfg: dict,
    available: pd.DataFrame | None = None,
) -> pypsa.Network:
    """Network over ``demand.index`` (hourly). ``demand`` has one column per region (MW).
    ``available`` is the sampled available fraction per unit (hours x unit_id); None means no
    forced outages."""
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

    # Thermal: by station and group.
    th = units[units["group"].isin(THERMAL)]
    for (region, station, group), g in th.groupby(["region", "station", "group"]):
        avail_mw = mw[g["unit_id"]].sum(axis=1)
        p_nom = float(avail_mw.max()) or float(g["capacity_mw"].sum())
        w = g["capacity_mw"]
        n.add(
            "Generator",
            f"{region} {station} {group}",
            bus=region,
            carrier=group,
            p_nom=p_nom,
            p_max_pu=avail_mw / p_nom,
            marginal_cost=float(np.average(g["srmc"], weights=w)),
            efficiency=float(3.6 / np.average(g["heat_rate"], weights=w))
            if g["heat_rate"].notna().all()
            else 1.0,
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
        inflow: float = 0.0,
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
    for region, g in units[units["group"] == "hydro"].groupby("region"):
        cf = d["hydro_capacity_factor"].get(region, 0.0)
        inflow = cf * float(g["capacity_mw"].sum())
        add_storage(
            f"{region} hydro",
            region,
            g,
            "hydro",
            1.0,
            d["hydro_storage_hours"],
            inflow=inflow,
            pump=False,
        )

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


def solve(n: pypsa.Network, cfg: dict, gas: GasConstraint | None = None) -> pypsa.Network:
    """Solve in consecutive windows of ``cfg['dispatch']['window_hours']`` hours."""
    hours = cfg["dispatch"]["window_hours"]
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
        day = xr.DataArray(
            days[winter], coords={"snapshot": sns[winter]}, dims="snapshot", name="day"
        )
        daily = (p * coef).sum(dim).groupby(day).sum()
        n.model.add_constraints(daily <= gas.cap_tj_per_day, name="southern_gas_daily")

    for start in range(0, len(n.snapshots), hours):
        sns = n.snapshots[start : start + hours]
        status, cond = n.optimize(
            snapshots=sns,
            solver_name="highs",
            solver_options={"output_flag": False, "threads": 1},
            extra_functionality=gas_cap if gas else None,
            include_objective_constant=False,
            progress=False,
        )
        if status != "ok":
            raise RuntimeError(f"window starting {sns[0]}: {status} ({cond})")
    return n


def daily_southern_gas_tj(n: pypsa.Network, regions: list[str]) -> pd.Series:
    gens = southern_gas_generators(n, regions)
    burn = n.generators_t.p[gens].mul(_gas_tj(n, gens), axis=1).sum(axis=1)
    return burn.groupby(burn.index.floor("D")).sum()
