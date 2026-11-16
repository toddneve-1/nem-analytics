"""AEMO aggregated price and demand data (5-minute RRP and total demand by region).

Source: AEMO's monthly "Aggregated price and demand data" CSV files, one per region per month::

    https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{YYYYMM}_{REGION}.csv

Columns: ``REGION, SETTLEMENTDATE, TOTALDEMAND, RRP, PERIODTYPE``. ``SETTLEMENTDATE`` is the
*end* of the 5-minute interval in NEM time (AEST, UTC+10). A monthly file covers intervals ending
from 00:05 on the 1st to 00:00 on the 1st of the next month.

Design choices:

* Each month is cached as its own parquet file, so downloads are resumable and re-runs are free.
* Values are stored exactly as published. Prices are **not** clipped: negative prices and prices
  at the market price cap are real market outcomes and are the point of the exercise.
* Data quality problems (duplicates, missing intervals, unparseable values) are detected and
  reported. Duplicates and out-of-month rows raise; missing intervals are reported, not filled.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from nemspike.timeutils import (
    INTERVAL,
    expected_interval_ends,
    months_between,
    nem_day_start_utc,
    nem_naive_to_utc,
)

log = logging.getLogger(__name__)

URL_TEMPLATE = (
    "https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{yyyymm}_{region}.csv"
)
RAW_COLUMNS = ["REGION", "SETTLEMENTDATE", "TOTALDEMAND", "RRP", "PERIODTYPE"]
COLUMNS = ["interval_start", "interval_end", "region", "rrp", "demand_mw", "period_type"]


class DataQualityError(ValueError):
    """Raised when downloaded data cannot be used safely."""


@dataclass
class QualityReport:
    """Summary of data checks for a price and demand table."""

    n_rows: int
    n_expected: int
    missing_interval_ends: list[pd.Timestamp] = field(default_factory=list)
    n_negative_price: int = 0
    n_above: dict[float, int] = field(default_factory=dict)
    price_min: float = float("nan")
    price_max: float = float("nan")

    @property
    def n_missing(self) -> int:
        return len(self.missing_interval_ends)

    def summary(self) -> str:
        above = ", ".join(f"> ${k:,.0f}: {v:,}" for k, v in self.n_above.items())
        return (
            f"rows {self.n_rows:,} of {self.n_expected:,} expected; missing {self.n_missing:,}; "
            f"price range ${self.price_min:,.2f} to ${self.price_max:,.2f}/MWh; "
            f"negative {self.n_negative_price:,}; {above}"
        )


def parse_price_demand_csv(source: str | bytes | Path) -> pd.DataFrame:
    """Parse one AEMO price and demand CSV into the project's standard schema.

    Returns columns ``interval_start``/``interval_end`` (UTC, tz-aware), ``region``, ``rrp``
    ($/MWh), ``demand_mw`` and ``period_type``.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    elif isinstance(source, str):
        source = io.StringIO(source)
    raw = pd.read_csv(source, dtype=str)

    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise DataQualityError(f"missing expected columns: {sorted(missing)}")

    naive_end = pd.to_datetime(raw["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
    rrp = pd.to_numeric(raw["RRP"], errors="coerce")
    demand = pd.to_numeric(raw["TOTALDEMAND"], errors="coerce")
    bad = naive_end.isna() | rrp.isna() | demand.isna()
    if bad.any():
        examples = raw.loc[bad].head(3).to_dict("records")
        raise DataQualityError(f"{int(bad.sum())} unparseable rows, e.g. {examples}")

    end = nem_naive_to_utc(naive_end)
    df = pd.DataFrame(
        {
            "interval_start": end - INTERVAL,
            "interval_end": end,
            "region": raw["REGION"].str.strip(),
            "rrp": rrp.astype("float64"),
            "demand_mw": demand.astype("float64"),
            "period_type": raw["PERIODTYPE"].str.strip(),
        }
    )
    return df.sort_values("interval_end", ignore_index=True)


def check_month_bounds(df: pd.DataFrame, year: int, month: int) -> None:
    """Raise if any row falls outside the month's interval-ending window."""
    lo = nem_day_start_utc(date(year, month, 1))
    hi = nem_day_start_utc((pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthBegin()).date())
    outside = (df["interval_end"] <= lo) | (df["interval_end"] > hi)
    if outside.any():
        raise DataQualityError(f"{int(outside.sum())} rows outside {year}-{month:02d}")


def check_duplicates(df: pd.DataFrame) -> None:
    """Raise on duplicated (region, interval_end) keys; never silently drop them."""
    dup = df.duplicated(["region", "interval_end"], keep=False)
    if dup.any():
        raise DataQualityError(
            f"{int(dup.sum())} duplicated rows, e.g. {df.loc[dup, 'interval_end'].head(3).tolist()}"
        )


def quality_report(
    df: pd.DataFrame, start: date, end: date, thresholds: tuple[float, ...] = (300.0, 1000.0)
) -> QualityReport:
    """Compare a table against the full expected 5-minute grid for ``start``..``end``."""
    expected = expected_interval_ends(start, end)
    present = pd.DatetimeIndex(df["interval_end"])
    missing = expected.difference(present)
    return QualityReport(
        n_rows=len(df),
        n_expected=len(expected),
        missing_interval_ends=list(missing),
        n_negative_price=int((df["rrp"] < 0).sum()),
        n_above={t: int((df["rrp"] > t).sum()) for t in thresholds},
        price_min=float(df["rrp"].min()),
        price_max=float(df["rrp"].max()),
    )


def _cache_path(raw_dir: Path, region: str, year: int, month: int) -> Path:
    return raw_dir / "aemo" / region / f"{year}-{month:02d}.parquet"


def _download(url: str, session: requests.Session, retries: int = 4, timeout: float = 60) -> bytes:
    """GET with exponential backoff on network errors and 5xx responses."""
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code < 500:
                resp.raise_for_status()
                return resp.content
            log.warning("HTTP %s from %s", resp.status_code, url)
        except requests.ConnectionError as exc:
            log.warning("connection error from %s: %s", url, exc)
        if attempt < retries:
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url} after {retries + 1} attempts")


def fetch_month(
    year: int,
    month: int,
    region: str,
    raw_dir: Path,
    session: requests.Session | None = None,
    force: bool = False,
) -> Path:
    """Download, validate and cache one month. Returns the parquet path."""
    path = _cache_path(raw_dir, region, year, month)
    if path.exists() and not force:
        return path
    url = URL_TEMPLATE.format(yyyymm=f"{year}{month:02d}", region=region)
    log.info("downloading %s", url)
    df = parse_price_demand_csv(_download(url, session or requests.Session()))
    if set(df["region"]) != {region}:
        raise DataQualityError(f"unexpected regions in {url}: {sorted(set(df['region']))}")
    check_month_bounds(df, year, month)
    check_duplicates(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def fetch_range(
    start: date,
    end: date,
    region: str,
    raw_dir: Path,
    session: requests.Session | None = None,
    force: bool = False,
) -> list[Path]:
    """Fetch every month overlapping ``start``..``end``."""
    session = session or requests.Session()
    return [
        fetch_month(y, m, region, raw_dir, session=session, force=force)
        for y, m in months_between(start, end)
    ]


def load_price_demand(start: date, end: date, region: str, raw_dir: Path) -> pd.DataFrame:
    """Load cached months and trim to NEM days ``start``..``end`` inclusive.

    Raises if any month is not cached (run ``fetch_range`` first) or if duplicates are found
    across months.
    """
    frames = []
    for y, m in months_between(start, end):
        path = _cache_path(raw_dir, region, y, m)
        if not path.exists():
            raise FileNotFoundError(f"{path} not cached; run `nemspike fetch-prices` first")
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    lo = nem_day_start_utc(start)
    hi = nem_day_start_utc(end) + pd.Timedelta(days=1)
    df = df[(df["interval_end"] > lo) & (df["interval_end"] <= hi)]
    check_duplicates(df)
    return df.sort_values("interval_end", ignore_index=True)[COLUMNS]
