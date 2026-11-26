"""Hourly demand and wind/solar availability from AEMO's ISP traces (2024 ISP format).

Traces are read from the processed parquet files published by the openISP project (mirrored from
AEMO, fetched with ``isp_trace_parser.remote.fetch_trace_data``). They are half-hourly, labelled
by interval end, for one weather reference year mapped onto future dates.

* **Demand:** ``OPSO_MODELLING`` (operational demand as modelled, sent out) by ISP sub-region,
  summed to the five NEM regions and averaged to hours. 10 % and 50 % probability of exceedance.
* **Wind and solar:** half-hourly capacity factors per project. Each unit in the fleet is matched
  to its own project trace by name; unmatched units take the capacity-weighted average of matched
  units of the same type in the region. The result is a capacity-weighted regional capacity factor
  per technology group, which is how the network represents wind and solar.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nemmodel.inputs import ModelYear

SUBREGION_TO_REGION = {
    "NQ": "QLD1",
    "CQ": "QLD1",
    "GG": "QLD1",
    "SQ": "QLD1",
    "NNSW": "NSW1",
    "CNSW": "NSW1",
    "SNW": "NSW1",
    "SNSW": "NSW1",
    "VIC": "VIC1",
    "CSA": "SA1",
    "SESA": "SA1",
    "TAS": "TAS1",
}
TRACE_TYPE = {"wind": ["WIND"], "solar": ["SAT", "FFP"]}
# Fallback when no unit of a group in a region has its own trace: the mean of that region's
# renewable energy zone traces (zone ids start with the region letter).
ZONE_PREFIX = {"QLD1": "Q", "NSW1": "N", "VIC1": "V", "SA1": "S", "TAS1": "T"}
ZONE_TYPE = {"wind": "WH", "solar": "SAT"}


def _period_filter(year: ModelYear) -> list[tuple]:
    start = pa.scalar(year.start, type=pa.timestamp("us"))
    end = pa.scalar(year.start + pd.DateOffset(years=1), type=pa.timestamp("us"))
    return [("datetime", ">", start), ("datetime", "<=", end)]


def _to_hourly(df: pd.DataFrame, value: str, by: list[str]) -> pd.DataFrame:
    """Half-hourly, interval-ending values to hourly means labelled by hour start."""
    hour = (df["datetime"] - pd.Timedelta(minutes=30)).dt.floor("h")  # interval end -> start
    return df.assign(hour=hour).groupby([*by, "hour"])[value].mean().unstack(by)


def hourly_demand(trace_dir: Path, year: ModelYear, scenario: str = "Step Change") -> pd.DataFrame:
    """Hourly operational demand (MW) by region and POE: columns like ``NSW1_POE50``."""
    files = glob.glob(f"{trace_dir}/demand/scenario={scenario}/**/*.parquet", recursive=True)
    if not files:
        raise FileNotFoundError(f"no demand traces for {scenario} under {trace_dir}")
    filt = [*_period_filter(year), ("demand_type", "=", "OPSO_MODELLING")]
    t = pa.concat_tables(
        [
            pq.read_table(f, filters=filt, columns=["datetime", "value", "subregion", "poe"])
            for f in files
        ]
    )
    df = t.to_pandas()
    df["region"] = df["subregion"].map(SUBREGION_TO_REGION)
    if df["region"].isna().any():
        raise ValueError(
            f"unmapped sub-regions: {sorted(df.loc[df.region.isna(), 'subregion'].unique())}"
        )
    regional = df.groupby(["datetime", "region", "poe"], as_index=False)["value"].sum()
    hourly = _to_hourly(regional, "value", ["region", "poe"])
    hourly.columns = [f"{r}_{p}" for r, p in hourly.columns]
    return hourly.sort_index(axis=1)


def _norm(name: str) -> str:
    s = re.sub(r"\b(wind|solar|farm|park|power|station|pv|project|stage)\b", " ", name.lower())
    return re.sub(r"[^a-z0-9]+", "", s)


def regional_vre_capacity_factors(
    trace_dir: Path, units: pd.DataFrame, year: ModelYear
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Capacity-weighted hourly capacity factor per region and group (``VIC1_wind`` ...), and a
    table recording how each wind/solar unit was matched to a trace."""
    files = glob.glob(f"{trace_dir}/project/**/*.parquet", recursive=True)
    t = pa.concat_tables(
        [
            pq.read_table(
                f,
                filters=_period_filter(year),
                columns=["datetime", "value", "project", "resource_type"],
            )
            for f in files
        ]
    )
    t = t.filter(pc.is_in(t["resource_type"], pa.array(TRACE_TYPE["wind"] + TRACE_TYPE["solar"])))
    df = t.to_pandas()
    project_cf = _to_hourly(df, "value", ["project"])  # hour x project
    by_norm = {_norm(p): p for p in project_cf.columns}

    vre = units[units["group"].isin(["wind", "solar"])].copy()
    vre["trace"] = [by_norm.get(_norm(s)) for s in vre["station"]]
    out, records = {}, []
    for (region, group), g in vre.groupby(["region", "group"]):
        matched = g[g["trace"].notna()]
        if matched.empty:
            out[f"{region}_{group}"] = _zone_cf(trace_dir, year, region, group)
            source, matched_mw = f"zone mean ({ZONE_TYPE[group]})", 0.0
        else:
            w = matched.groupby("trace")["capacity_mw"].sum()
            # unmatched units take the capacity-weighted average of matched units
            out[f"{region}_{group}"] = project_cf[w.index].mul(w, axis=1).sum(axis=1) / w.sum()
            source, matched_mw = "project traces", float(w.sum())
        records.append(
            {
                "region": region,
                "group": group,
                "source": source,
                "units": len(g),
                "matched_units": len(matched),
                "matched_mw": matched_mw,
                "total_mw": float(g["capacity_mw"].sum()),
            }
        )
    return pd.DataFrame(out).sort_index(axis=1), pd.DataFrame(records)


def _zone_cf(trace_dir: Path, year: ModelYear, region: str, group: str) -> pd.Series:
    files = glob.glob(f"{trace_dir}/zone/**/*.parquet", recursive=True)
    t = pa.concat_tables(
        [
            pq.read_table(
                f,
                filters=_period_filter(year),
                columns=["datetime", "value", "zone", "resource_type"],
            )
            for f in files
        ]
    )
    df = t.to_pandas()
    df = df[
        (df["resource_type"] == ZONE_TYPE[group]) & df["zone"].str.startswith(ZONE_PREFIX[region])
    ]
    if df.empty:
        raise ValueError(f"no traces for {group} in {region}")
    return _to_hourly(df, "value", ["zone"]).mean(axis=1)
