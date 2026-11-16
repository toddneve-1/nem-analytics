from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from nemspike.data import aemo
from nemspike.data.aemo import (
    DataQualityError,
    check_duplicates,
    check_month_bounds,
    fetch_month,
    load_price_demand,
    parse_price_demand_csv,
    quality_report,
)

HEADER = "REGION,SETTLEMENTDATE,TOTALDEMAND,RRP,PERIODTYPE\n"


def make_month_csv(year: int, month: int, drop: int | None = None, price: float = 50.0) -> str:
    """Synthetic AEMO file covering intervals ending 00:05 on the 1st to 00:00 next month."""
    start = pd.Timestamp(year=year, month=month, day=1)
    ends = pd.date_range(
        start + pd.Timedelta(minutes=5), start + pd.offsets.MonthBegin(), freq="5min"
    )
    rows = [
        f"VIC1,{t:%Y/%m/%d %H:%M:%S},{4000 + i % 100:.2f},{price - (i % 7) * 30:.2f},TRADE"
        for i, t in enumerate(ends)
        if i != drop
    ]
    return HEADER + "\n".join(rows) + "\n"


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Stands in for requests.Session; serves synthetic months and counts calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, url: str, timeout: float) -> FakeResponse:
        self.calls.append(url)
        yyyymm = url.split("PRICE_AND_DEMAND_")[1][:6]
        return FakeResponse(make_month_csv(int(yyyymm[:4]), int(yyyymm[4:])).encode())


def test_parse_converts_interval_end_to_utc_and_derives_start() -> None:
    df = parse_price_demand_csv(HEADER + "VIC1,2024/01/01 00:05:00,3977.1,51.80,TRADE\n")
    row = df.iloc[0]
    assert row["interval_end"] == pd.Timestamp("2023-12-31 14:05", tz="UTC")
    assert row["interval_start"] == pd.Timestamp("2023-12-31 14:00", tz="UTC")
    assert row["rrp"] == pytest.approx(51.80)
    assert row["demand_mw"] == pytest.approx(3977.1)


def test_parse_keeps_negative_and_extreme_prices_unclipped() -> None:
    text = HEADER + (
        "VIC1,2024/01/01 00:05:00,4000,-1000.00,TRADE\n"
        "VIC1,2024/01/01 00:10:00,4000,17500.00,TRADE\n"
    )
    assert parse_price_demand_csv(text)["rrp"].tolist() == [-1000.0, 17500.0]


def test_parse_rejects_unparseable_values() -> None:
    with pytest.raises(DataQualityError):
        parse_price_demand_csv(HEADER + "VIC1,2024/01/01 00:05:00,4000,abc,TRADE\n")


def test_parse_rejects_missing_columns() -> None:
    with pytest.raises(DataQualityError):
        parse_price_demand_csv("REGION,SETTLEMENTDATE,RRP\nVIC1,2024/01/01 00:05:00,1\n")


def test_full_month_has_expected_rows_and_passes_bounds() -> None:
    df = parse_price_demand_csv(make_month_csv(2024, 1))
    assert len(df) == 31 * 288
    check_month_bounds(df, 2024, 1)


def test_month_bounds_reject_row_from_other_month() -> None:
    df = parse_price_demand_csv(make_month_csv(2024, 1))
    with pytest.raises(DataQualityError):
        check_month_bounds(df, 2024, 2)


def test_duplicates_raise_rather_than_being_dropped() -> None:
    df = parse_price_demand_csv(make_month_csv(2024, 1))
    with pytest.raises(DataQualityError):
        check_duplicates(pd.concat([df, df.iloc[[10]]]))


def test_quality_report_detects_missing_interval() -> None:
    df = parse_price_demand_csv(make_month_csv(2024, 1, drop=100))
    report = quality_report(df, date(2024, 1, 1), date(2024, 1, 31))
    assert report.n_expected == 31 * 288
    assert report.n_missing == 1
    assert report.missing_interval_ends[0] == df["interval_end"].iloc[99] + aemo.INTERVAL


def test_fetch_month_caches_and_skips_second_download(tmp_path: Path) -> None:
    session = FakeSession()
    p1 = fetch_month(2024, 1, "VIC1", tmp_path, session=session)  # type: ignore[arg-type]
    p2 = fetch_month(2024, 1, "VIC1", tmp_path, session=session)  # type: ignore[arg-type]
    assert p1 == p2 and p1.exists()
    assert len(session.calls) == 1


def test_load_trims_to_nem_days_across_month_boundary(tmp_path: Path) -> None:
    session = FakeSession()
    for m in (1, 2):
        fetch_month(2024, m, "VIC1", tmp_path, session=session)  # type: ignore[arg-type]
    df = load_price_demand(date(2024, 1, 31), date(2024, 2, 1), "VIC1", tmp_path)
    assert len(df) == 2 * 288
    assert df["interval_end"].is_monotonic_increasing
    assert df["interval_end"].iloc[0] == pd.Timestamp("2024-01-30 14:05", tz="UTC")


def test_load_requires_cache(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_price_demand(date(2024, 1, 1), date(2024, 1, 2), "VIC1", tmp_path)


@pytest.mark.network
def test_live_download_one_month(tmp_path: Path) -> None:
    path = fetch_month(2024, 1, "VIC1", tmp_path)
    df = pd.read_parquet(path)
    assert len(df) == 31 * 288
