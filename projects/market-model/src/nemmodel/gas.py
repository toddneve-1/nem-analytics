"""Southern winter gas balance: how much gas power stations can burn each winter day.

    allowance(day) = southern production + northern supply + storage withdrawal
                     - other demand(day)

Other demand (homes, commerce, industry) is higher on colder days. Each day is placed between
the GSOO's average winter day and peak day using southern electricity demand as a proxy for how
cold it is: the coldest-demand day gets the peak figure, an average day the average figure.
This links gas shortage to the same cold, dark, still days that strain the power system.

Production, northern supply and other demand must come from the GSOO
(``config/gas_balance.yaml``); storage withdrawal comes from the workbook.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

REQUIRED = (
    "southern_production_tj_d",
    "northern_supply_tj_d",
    "other_demand_peak_tj_d",
    "other_demand_average_tj_d",
)


class MissingGasDataError(ValueError):
    """The GSOO values the balance needs have not been filled in."""


def load_balance(path: Path) -> dict:
    """Read the balance and check every GSOO value is present."""
    b = yaml.safe_load(path.read_text())
    missing = [f"winters.{w}.{k}" for w, v in b["winters"].items() for k in REQUIRED
               if v.get(k) is None]  # fmt: off
    if b.get("source") is None:
        missing.insert(0, "source")
    if missing:
        raise MissingGasDataError(
            f"{path.name} is missing GSOO values: {', '.join(missing)}. "
            "See docs/data_needed.md; the illustrative cap is used until these are filled in."
        )
    return b


def storage_withdrawal_tj_d(infrastructure: pd.DataFrame) -> float:
    """Total withdrawal capacity of online southern gas storage (TJ/day)."""
    s = infrastructure[(infrastructure["kind"] == "storage") & infrastructure["southern"]
                       & (infrastructure["status"] == "Online")]  # fmt: off
    return float(s["capacity_tj_d"].sum())


def daily_allowance(
    balance: dict,
    infrastructure: pd.DataFrame,
    southern_demand_mw: pd.Series,
    months: list[int],
) -> pd.Series:
    """Gas available to southern power stations (TJ) on each winter day.

    ``southern_demand_mw`` is hourly electricity demand summed over the southern regions; its
    daily mean ranks how cold each day is.
    """
    default_storage = balance.get("storage_withdrawal_tj_d")
    daily = southern_demand_mw.groupby(southern_demand_mw.index.floor("D")).mean()
    daily = daily[daily.index.month.isin(months)]
    out = {}
    for year, d in daily.groupby(daily.index.year):
        w = balance["winters"][year]
        # 0 on an average winter day, 1 on the highest-demand day; below-average days use the
        # average figure (not less), which keeps the allowance from exceeding supply on mild days.
        cold = ((d - d.mean()) / (d.max() - d.mean())).clip(lower=0)
        other = w["other_demand_average_tj_d"] + cold * (
            w["other_demand_peak_tj_d"] - w["other_demand_average_tj_d"]
        )
        # Storage withdrawal: per winter if given, else one figure for all, else the workbook.
        storage = (w.get("storage_withdrawal_tj_d") or default_storage
                   or storage_withdrawal_tj_d(infrastructure))  # fmt: off
        supply = w["southern_production_tj_d"] + w["northern_supply_tj_d"] + storage
        out[year] = supply - other
    return pd.concat(out.values()).sort_index()
