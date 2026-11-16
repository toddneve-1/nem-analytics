from datetime import date

import pandas as pd
import pytest

from nemspike.timeutils import (
    expected_interval_ends,
    months_between,
    nem_naive_to_utc,
    to_nem_time,
)


def test_nem_time_is_utc_plus_ten_in_summer_and_winter() -> None:
    naive = pd.Series(pd.to_datetime(["2024-01-15 12:00", "2024-07-15 12:00"]))
    utc = nem_naive_to_utc(naive)
    # Both seasons map to 02:00 UTC: there is no daylight-saving shift in NEM time.
    assert list(utc.dt.hour) == [2, 2]


def test_melbourne_civil_time_would_be_wrong_in_summer() -> None:
    """Documents the bug we avoid: Australia/Melbourne is UTC+11 in January."""
    naive = pd.Series(pd.to_datetime(["2024-01-15 12:00"]))
    wrong = naive.dt.tz_localize("Australia/Melbourne").dt.tz_convert("UTC")
    right = nem_naive_to_utc(naive)
    assert (right - wrong).iloc[0] == pd.Timedelta(hours=1)


def test_round_trip() -> None:
    naive = pd.Series(pd.date_range("2024-03-01", periods=10, freq="5min"))
    back = to_nem_time(nem_naive_to_utc(naive)).dt.tz_localize(None)
    pd.testing.assert_series_equal(back, naive)


def test_rejects_already_aware_input() -> None:
    aware = pd.Series(pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC"))
    with pytest.raises(ValueError):
        nem_naive_to_utc(aware)


def test_expected_interval_ends_one_day() -> None:
    idx = expected_interval_ends(date(2024, 1, 1), date(2024, 1, 1))
    assert len(idx) == 288
    assert str(idx.tz) == "UTC"
    # First interval of the NEM day ends 00:05 AEST = 14:05 UTC the previous day.
    assert idx[0] == pd.Timestamp("2023-12-31 14:05", tz="UTC")
    # Last interval ends at midnight AEST (labelled as the next calendar day).
    assert idx[-1] == pd.Timestamp("2024-01-01 14:00", tz="UTC")


def test_months_between() -> None:
    assert months_between(date(2023, 11, 15), date(2024, 2, 1)) == [
        (2023, 11),
        (2023, 12),
        (2024, 1),
        (2024, 2),
    ]
