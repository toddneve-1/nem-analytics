"""Time handling for NEM data.

Conventions used throughout the project:

* **Market time** is fixed Australian Eastern Standard Time (AEST, UTC+10) all year round. The NEM
  does not observe daylight saving, even though Victoria's civil time does. Using the IANA zone
  ``Australia/Melbourne`` would shift every summer interval by one hour, so we never use it for
  market timestamps.
* **Storage** is time-zone-aware UTC. Weather APIs return UTC, so storing everything in UTC makes
  joins unambiguous.
* **Interval labelling**: AEMO's ``SETTLEMENTDATE`` labels each 5-minute dispatch interval by its
  *end* time. ``2024/01/01 00:05:00`` is the interval 00:00-00:05 AEST. We keep both
  ``interval_end`` (AEMO's label) and ``interval_start``.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

# Note the inverted sign in POSIX-style Etc zones: "Etc/GMT-10" means UTC+10.
NEM_TZ = "Etc/GMT-10"
UTC = "UTC"
INTERVAL = pd.Timedelta(minutes=5)


def nem_naive_to_utc(values: pd.Series | pd.DatetimeIndex) -> pd.Series | pd.DatetimeIndex:
    """Interpret naive timestamps as NEM time (UTC+10) and convert them to UTC."""
    if isinstance(values, pd.Series):
        if values.dt.tz is not None:
            raise ValueError("expected naive timestamps; got time-zone-aware values")
        return values.dt.tz_localize(NEM_TZ).dt.tz_convert(UTC)
    if values.tz is not None:
        raise ValueError("expected naive timestamps; got time-zone-aware values")
    return values.tz_localize(NEM_TZ).tz_convert(UTC)


def to_nem_time(values: pd.Series | pd.DatetimeIndex) -> pd.Series | pd.DatetimeIndex:
    """Convert time-zone-aware timestamps to NEM time (UTC+10) for display and calendar use."""
    if isinstance(values, pd.Series):
        return values.dt.tz_convert(NEM_TZ)
    return values.tz_convert(NEM_TZ)


def nem_day_start_utc(d: date) -> pd.Timestamp:
    """00:00 NEM time on the given date, expressed in UTC."""
    return pd.Timestamp(d).tz_localize(NEM_TZ).tz_convert(UTC)


def months_between(start: date, end: date) -> list[tuple[int, int]]:
    """All (year, month) pairs from the month of ``start`` to the month of ``end`` inclusive."""
    periods = pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M")
    return [(p.year, p.month) for p in periods]


def expected_interval_ends(start: date, end: date) -> pd.DatetimeIndex:
    """Every 5-minute interval-end timestamp (UTC) for NEM days ``start``..``end`` inclusive.

    A NEM day ``d`` covers intervals ending from ``d 00:05`` to ``d+1 00:00`` AEST.
    """
    first = nem_day_start_utc(start) + INTERVAL
    last = nem_day_start_utc(end) + pd.Timedelta(days=1)
    return pd.date_range(first, last, freq=INTERVAL)
