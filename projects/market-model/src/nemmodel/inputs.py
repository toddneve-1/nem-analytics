"""Model inputs from AEMO's ISP Inputs and Assumptions workbook.

Reads the workbook with ``isp-workbook-parser`` and produces tidy, versioned tables for one
financial year and scenario:

* ``units``: every generating and storage unit in the fleet for that year, with capacity,
  seasonal ratings, outage parameters, heat rate, variable O&M and fuel price;
* ``interconnectors``: inter-regional transfer limits by direction and season, including
  committed and anticipated uplifts in service by the start of the year.

Fleet definition (as in AEMO's ESOO reliability forecasts): **existing, committed and anticipated**
projects only, with no hypothetical new builds. "Additional policy-supported" projects are
excluded. A committed or anticipated unit counts if commissioned before the year starts; a unit
counts as retired if its expected closure (calendar) year is before the year ends.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REGION = {"NSW": "NSW1", "QLD": "QLD1", "VIC": "VIC1", "SA": "SA1", "TAS": "TAS1"}
INCLUDED_STATUS = ("Existing", "Committed", "Anticipated")

# Technology groups used in the network (units are aggregated by region and group).
TECH_GROUP = {
    "Steam Sub Critical": "coal",
    "Steam Super Critical": "coal",
    "CCGT": "ccgt",
    "CCGT - Gas Turbine": "ccgt",
    "CCGT - Steam Turbine": "ccgt",
    "OCGT (large GT)": "ocgt",
    "OCGT (small GT)": "ocgt",
    "Reciprocating engine": "recip",
    "Hydro": "hydro",
    "Wind": "wind",
    "Large scale Solar PV": "solar",
    "Biomass": "biomass",
}
FUEL_PRICE_TABLE = {
    "Gas": "gas_prices_existing_generators",
    "Black Coal": "coal_fuel_price",
    "Brown Coal": "coal_fuel_price",
    "Liquid Fuel": "liquid_fuel_prices",
}
# Interconnectors between the five regions, as sub-regional flow paths in the workbook.
# Forward direction in the workbook is from the first sub-region to the second.
FLOW_PATHS = {
    "NNSW-SQ": ("NSW1", "QLD1"),
    "NNSW-SQ (Terranora)": ("NSW1", "QLD1"),
    "SNSW-CSA": ("NSW1", "SA1"),
    "WNV-SNSW": ("VIC1", "NSW1"),
    "TAS-SEV": ("TAS1", "VIC1"),
    "WNV-SESA": ("VIC1", "SA1"),
    "WNV-CSA (Murraylink)": ("VIC1", "SA1"),
}


@dataclass(frozen=True)
class ModelYear:
    """A financial year, e.g. ``ModelYear(2029)`` is FY2028-29 (1 July 2028 to 30 June 2029)."""

    end_year: int

    @property
    def label(self) -> str:
        return f"{self.end_year - 1}-{str(self.end_year)[-2:]}"

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(f"{self.end_year - 1}-07-01")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _year_col(df: pd.DataFrame, year: ModelYear) -> str:
    if year.label not in df.columns:
        raise KeyError(f"{year.label} not in table columns")
    return year.label


def _fuel_prices(parser, scenario: str, year: ModelYear) -> pd.Series:
    """Fuel price ($/GJ) by power station for the scenario and year."""
    out = {}
    for fuel, table in FUEL_PRICE_TABLE.items():
        df = parser.get_table(table)
        scen_col = "Scenario" if "Scenario" in df.columns else "Gas price scenario"
        df = df[df[scen_col] == scenario]
        for name, price in zip(df["Generator"], df[_year_col(df, year)], strict=True):
            out[(fuel, name)] = float(price)
    return pd.Series(out)


def load_units(parser, year: ModelYear, scenario: str = "Step Change") -> pd.DataFrame:
    """Unit table for the year (see module docstring for the fleet definition)."""
    cap = parser.get_table("maximum_capacity_existing_committed_anticipated_additional_generators")
    summ = parser.get_table("existing_committed_anticipated_additional_generator_summary")
    summ = summ.rename(columns={"IASR ID / DLT names": "IASR ID"})
    closures = parser.get_table("expected_closure_years").rename(
        columns={"Expected Closure Year (Calendar year)": "closure_year"}
    )
    heat = parser.get_table("heat_rates_existing_committed_anticipated_additional_generators")
    vom = parser.get_table("variable_opex_existing_committed_anticipated_additional_generators")
    ratings = parser.get_table(
        "seasonal_ratings_existing_committed_anticipated_additional_generators"
    ).rename(columns={"_IASR ID": "IASR ID"})
    outages = parser.get_table("other_outages_existing_generators")

    df = cap.rename(
        columns={
            "Installed capacity (MW)": "capacity_mw",
            "Storage Capacity (MWh)": "storage_mwh",
            "Commissioning date": "commissioning",
        }
    )[
        [
            "IASR ID",
            "Power Station",
            "Status",
            "Technology",
            "Region",
            "capacity_mw",
            "storage_mwh",
            "commissioning",
        ]
    ]
    df = df.merge(
        summ[
            [
                "IASR ID",
                "Fuel type",
                "Fuel cost mapping",
                "Forced outage rate_Full outage (% of time)",
            ]
        ],
        on="IASR ID",
        how="left",
    ).rename(columns={"Forced outage rate_Full outage (% of time)": "outage_group"})
    df = df.merge(closures[["IASR ID", "closure_year"]], on="IASR ID", how="left")
    df = df.merge(heat[["IASR ID", "Heat rate (GJ/MWh)"]], on="IASR ID", how="left")
    vom_col = [c for c in vom.columns if c.startswith("Variable OPEX")][0]
    df = df.merge(vom[["IASR ID", vom_col]], on="IASR ID", how="left")
    df = df.rename(columns={"Heat rate (GJ/MWh)": "heat_rate", vom_col: "vom"})

    # Minimum stable levels (coal: the typical lowest band they run at; gas: the workbook's
    # single figure) and ramp rates. "Assumed sufficiently high" ramp rates become blank (no
    # limit).
    coal_msl = parser.get_table("coal_minimum_stable_level")
    coal_msl = coal_msl[["IASR ID", COAL_MSL_COLUMN]].rename(columns={COAL_MSL_COLUMN: "msl_mw"})
    gpg_msl = parser.get_table("gpg_min_stable_level_existing_generators")
    gpg_msl = gpg_msl[["IASR ID", "Min Stable Level (MW)"]].rename(
        columns={"Min Stable Level (MW)": "msl_mw"}
    )
    msl = pd.concat([coal_msl, gpg_msl]).drop_duplicates("IASR ID")
    df = df.merge(msl, on="IASR ID", how="left")
    ramps = parser.get_table("ramp_rates_existing_generators").rename(
        columns={"Max Ramp Up (MW/min)": "ramp_up_mw_min",
                 "Max Ramp Down (MW/min)": "ramp_down_mw_min"}
    )  # fmt: off
    df = df.merge(ramps[["IASR ID", "ramp_up_mw_min", "ramp_down_mw_min"]], on="IASR ID",
                  how="left")  # fmt: off
    for c in ("msl_mw", "ramp_up_mw_min", "ramp_down_mw_min"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Seasonal ratings for the year: summer typical / summer peak for the FY, and winter for the
    # calendar winters inside it (2028 for Jul-Sep 2028, 2029 for Apr-Jun 2029).
    y = year.end_year
    rating_cols = {
        f"Summer Typical_{year.label}": "rating_summer_typical",
        f"Summer Peak_{year.label}": "rating_summer_peak",
        f"Winter_{y - 1}": "rating_winter_early",
        f"Winter_{y}": "rating_winter_late",
    }
    available = {k: v for k, v in rating_cols.items() if k in ratings.columns}
    df = df.merge(
        ratings[["IASR ID", *available]].rename(columns=available), on="IASR ID", how="left"
    )

    # Outage parameters by outage group for the year.
    prop = {
        "Full outage (% of time)": "for_full",
        "Partial outage (% of time)": "for_partial",
        "Partial Outage Derating Factor (%)": "partial_derate",
        "Full outage MTTR (hrs)": "mttr_full",
        "Partial outage MTTR (hrs)": "mttr_partial",
    }
    o = outages[outages["Property"].isin(prop)].pivot_table(
        index="Fuel type", columns="Property", values=_year_col(outages, year), aggfunc="first"
    )
    o = o.rename(columns=prop)
    df = df.merge(o, left_on="outage_group", right_index=True, how="left")
    for c in ("for_full", "for_partial", "partial_derate"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0) / 100.0
    for c in ("mttr_full", "mttr_partial"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fleet for the year.
    df = df[df["Status"].isin(INCLUDED_STATUS)].copy()
    commissioning = pd.to_datetime(df["commissioning"], errors="coerce")
    df = df[commissioning.isna() | (commissioning < year.start)]
    closure = pd.to_numeric(df["closure_year"], errors="coerce")
    df = df[closure.isna() | (closure >= y)]  # closing in calendar year y or later counts

    df["region"] = df["Region"].map(REGION)
    df["group"] = df["Technology"].map(TECH_GROUP)
    df.loc[df["Technology"].str.startswith("Battery", na=False), "group"] = "battery"
    df.loc[df["Technology"].str.startswith("Pumped Hydro", na=False), "group"] = "pumped_hydro"
    df.loc[(df["group"] == "ocgt") & (df["Fuel type"] == "Liquid Fuel"), "group"] = "liquid"
    df.loc[(df["group"] == "recip") & (df["Fuel type"] == "Liquid Fuel"), "group"] = "liquid"

    prices = _fuel_prices(parser, scenario, year)
    df["fuel_price"] = [
        prices.get((f, m), np.nan)
        for f, m in zip(df["Fuel type"], df["Fuel cost mapping"], strict=True)
    ]
    for c in ("heat_rate", "vom", "capacity_mw", "storage_mwh", "fuel_price"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Wivenhoe and Shoalhaven are pumped storage but labelled "Hydro": storage energy marks them.
    df.loc[(df["group"] == "hydro") & (df["storage_mwh"] > 0), "group"] = "pumped_hydro"
    df["srmc"] = df["fuel_price"].fillna(0) * df["heat_rate"].fillna(0) + df["vom"].fillna(0)
    df["closure_year"] = pd.to_numeric(df["closure_year"], errors="coerce")
    cols = [
        "IASR ID",
        "Power Station",
        "Status",
        "Technology",
        "Fuel type",
        "region",
        "group",
        "capacity_mw",
        "storage_mwh",
        "closure_year",
        "heat_rate",
        "vom",
        "fuel_price",
        "srmc",
        "msl_mw",
        "ramp_up_mw_min",
        "ramp_down_mw_min",
        "for_full",
        "for_partial",
        "partial_derate",
        "mttr_full",
        "mttr_partial",
        *[c for c in rating_cols.values() if c in df.columns],
    ]
    out = df[cols].rename(
        columns={
            "IASR ID": "unit_id",
            "Power Station": "station",
            "Technology": "technology",
            "Fuel type": "fuel",
            "Status": "status",
        }
    )
    missing = out["group"].isna()
    if missing.any():
        raise ValueError(
            f"unmapped technologies: {sorted(out.loc[missing, 'technology'].unique())}"
        )
    return out.reset_index(drop=True)


def load_interconnectors(parser, year: ModelYear) -> pd.DataFrame:
    """Transfer limits (MW) between regions by direction and season, with committed/anticipated
    uplifts whose service date is before the year starts."""
    ic = parser.get_table("interconnector_transfer_capability")
    cols = {c: c for c in ic.columns}
    fwd_s = [c for c in cols if c.startswith("Forward") and "Summer typical" in c][0]
    fwd_w = [c for c in cols if c.startswith("Forward") and "Winter" in c][0]
    rev_s = [c for c in cols if c.startswith("Reverse") and "Summer typical" in c][0]
    rev_w = [c for c in cols if c.startswith("Reverse") and "Winter" in c][0]
    rows = []
    for _, r in ic.iterrows():
        path = str(r["Flow Paths"]).strip()
        if path not in FLOW_PATHS:
            continue
        a, b = FLOW_PATHS[path]
        rows.append(
            {
                "flow_path": path,
                "from": a,
                "to": b,
                "fwd_summer": float(r[fwd_s]),
                "fwd_winter": float(r[fwd_w]),
                "rev_summer": float(r[rev_s]),
                "rev_winter": float(r[rev_w]),
            }
        )
    df = pd.DataFrame(rows)

    timing = parser.get_table("network_committed_and_anticipated_project_timing")
    in_service = {
        str(p).strip()
        for p, d in zip(timing["Project"], pd.to_datetime(timing["Service date"]), strict=True)
        if d < year.start
    }
    # The uplift table is cumulative across its project columns (each column repeats the running
    # total after successive projects) and some cells hold labels rather than numbers. For each
    # inter-regional path, take the largest numeric uplift among projects in service.
    uplift = parser.get_table("notional_transfer_uplift_committed_and_anticipated")
    path_col = uplift.columns[0]
    df["uplift_note"] = ""
    for _, r in uplift.iterrows():
        label = str(r[path_col]).split(" (")[0].replace(" – ", "-").strip()  # noqa: RUF001
        if label not in df["flow_path"].to_numpy():
            continue  # intra-regional path: not represented in a five-region model
        for key, word in (("fwd", "Forward"), ("rev", "Reverse")):
            vals = [
                pd.to_numeric(r[c], errors="coerce")
                for c in uplift.columns[1:]
                if word in c and any(c.startswith(p) for p in in_service)
            ]
            vals = [v for v in vals if pd.notna(v) and v > 0]
            if vals:
                mask = df["flow_path"] == label
                df.loc[mask, f"{key}_summer"] += max(vals)
                df.loc[mask, f"{key}_winter"] += max(vals)
                df.loc[mask, "uplift_note"] += f"{key} +{max(vals):g} (committed projects); "
    return df


COAL_MSL_COLUMN = "Minimum Stable Level (MW)_Typical Lowest Band"

# Monthly hydro inflow tables used by the model, and their units in the workbook.
HYDRO_INFLOW_TABLES = {
    "tasmania": ("hydro_inflows_tas", "GWh"),
    "snowy": ("hydro_inflows_snowy", "GL (representative; shape only)"),
}
MONTHS = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]


def load_hydro_inflows(parser) -> pd.DataFrame:
    """Monthly inflows by scheme and weather reference year (financial year ending).

    Tasmania is given in GWh for the whole Hydro Tasmania scheme. Snowy is given in GL and AEMO
    describes it as representative of weather variability only, so the model uses its monthly
    shape, not its level.
    """
    rows = []
    for scheme, (table, unit) in HYDRO_INFLOW_TABLES.items():
        t = parser.get_table(table)
        ref = pd.to_numeric(t["Reference Year (FYE)"], errors="coerce")
        for year, (_, r) in zip(ref, t.iterrows(), strict=True):
            if pd.isna(year):
                continue  # e.g. the "2007 (Dry)" sensitivity row
            for m in MONTHS:
                rows.append({"scheme": scheme, "unit": unit, "reference_year": int(year),
                             "month": m, "inflow": float(r[m])})  # fmt: off
    return pd.DataFrame(rows)


SOUTHERN_STATES = ("NSW", "VIC", "SA", "TAS")


def load_gas_infrastructure(parser) -> pd.DataFrame:
    """Gas processing plants, storage and pipelines from the workbook (2024 and 2025 GSOO data).

    One row per facility with its kind, status, location and daily capacity (TJ/day). Pipeline
    capacities are often given per direction ("565 South 195 North"); the text is kept as
    published in ``capacity_text`` and the first number is parsed into ``capacity_tj_d``.
    ``southern`` marks facilities located in NSW, VIC, SA or TAS (a pipeline touching one counts).
    """
    plants = parser.get_table("gas_system_properties_processing_facilities")
    storage = parser.get_table("gas_system_properties_storage_facilities")
    pipes = parser.get_table("gas_system_properties_pipelines")
    cap_col = "Capacity (TJ per day)"
    rows = []
    for _, r in plants.iterrows():
        rows.append({"kind": "processing", "name": r["Name"], "status": r["Status"],
                     "location": r["Location"], "capacity_text": str(r[cap_col]),
                     "storage_pj": np.nan, "source": r["Source"]})  # fmt: off
    for _, r in storage.iterrows():
        rows.append({"kind": "storage", "name": r["Storage Facility Name"], "status": r["Status"],
                     "location": r["Location"],
                     "capacity_text": str(r["Withdrawl Rate from Storage (TJ/day)"]),
                     "storage_pj": pd.to_numeric(str(r["Storage Capacity (PJ)"]).split()[-1],
                                                 errors="coerce"),
                     "source": r["Source"]})  # fmt: off
    for _, r in pipes.iterrows():
        rows.append({"kind": "pipeline", "name": r["Name"], "status": r["Status"],
                     "location": r["Location"], "capacity_text": str(r[cap_col]),
                     "storage_pj": np.nan, "source": r["Source"]})  # fmt: off
    df = pd.DataFrame(rows)
    df["capacity_tj_d"] = pd.to_numeric(
        df["capacity_text"].str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0], errors="coerce"
    )
    loc = df["location"].astype(str).str.upper()
    df["southern"] = loc.apply(lambda x: any(st in x for st in SOUTHERN_STATES))
    # Storage locations are free text (e.g. "Port Campbell, Victoria").
    df.loc[loc.str.contains("VICTORIA|NEW SOUTH WALES|TOMAGO"), "southern"] = True
    return df


def write_inputs(
    workbook: Path, out_dir: Path, year: ModelYear, scenario: str, source_url: str
) -> dict:
    """Parse the workbook and write versioned input tables plus a manifest."""
    from isp_workbook_parser import Parser

    parser = Parser(str(workbook))
    out_dir.mkdir(parents=True, exist_ok=True)
    units = load_units(parser, year, scenario)
    ics = load_interconnectors(parser, year)
    units.to_csv(out_dir / "units.csv", index=False)
    ics.to_csv(out_dir / "interconnectors.csv", index=False)
    load_hydro_inflows(parser).to_csv(out_dir / "hydro_inflows.csv", index=False)
    load_gas_infrastructure(parser).to_csv(out_dir / "gas_infrastructure.csv", index=False)
    manifest = {
        "workbook": {
            "url": source_url,
            "sha256": sha256(workbook),
            "version": str(parser.workbook_version),
        },
        "financial_year": year.label,
        "scenario": scenario,
        "fleet": "existing, committed and anticipated (ESOO definition)",
        "units": len(units),
        "capacity_mw_by_region_group": units.groupby(["region", "group"])["capacity_mw"]
        .sum()
        .round(1)
        .unstack(fill_value=0)
        .to_dict(orient="index"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
