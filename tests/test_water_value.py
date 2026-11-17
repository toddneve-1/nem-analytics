from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from nemspike.battery.lp import BatterySpec, optimise
from nemspike.battery.water_value import storage_values

SPEC = BatterySpec(100.0, 200.0, 0.9, 20.0, 180.0, 100.0, degradation_cost_per_mwh=40.0)


def _net(prices: np.ndarray, spec: BatterySpec, s0: float) -> float:
    d = optimise(prices, spec, soc_start=s0, soc_end=None, method="lp")
    return d.revenue(prices) - d.degradation_cost(spec)


def test_first_interval_value_matches_finite_difference() -> None:
    """The dual at t=0 is d(optimal revenue)/d(initial energy); check against a bump."""
    rng = np.random.default_rng(0)
    prices = rng.normal(80, 60, 96)
    values = storage_values(prices, SPEC, soc_end=None)
    bump = 1.0
    fd = (_net(prices, SPEC, 100.0 + bump) - _net(prices, SPEC, 100.0)) / bump
    assert values[0] == pytest.approx(fd, rel=1e-3, abs=0.5)


def test_value_is_replacement_cost_when_charging_is_easy() -> None:
    """With two hours of cheap prices before the peak the battery fills anyway, so an extra MWh
    just means buying one less: worth the cheap price divided by charging efficiency."""
    prices = np.array([30.0] * 24 + [300.0] * 12 + [30.0] * 24)
    values = storage_values(prices, SPEC, soc_end=None)
    assert values[10] == pytest.approx(30.0 / SPEC.eta_one_way, rel=1e-3)
    # After the peak only $30 prices remain, below the cost of losses and wear: worth nothing.
    assert values[50] == pytest.approx(0.0, abs=1e-6)


def test_value_is_peak_value_when_charging_is_constrained() -> None:
    """With only 10 minutes of cheap power before the peak, charging is power-limited, so an extra
    stored MWh is sold at the peak: worth the peak price after discharge losses and wear."""
    prices = np.array([30.0] * 2 + [300.0] * 12 + [30.0] * 12)
    values = storage_values(prices, SPEC, soc_end=None)
    eta, k = SPEC.eta_one_way, SPEC.degradation_cost_per_mwh
    assert values[1] == pytest.approx((300.0 - k) * eta, rel=1e-3)


def test_negative_price_loophole_is_removed_before_taking_duals() -> None:
    """With deeply negative prices and no wear cost the plain LP burns energy; the water value
    must come from the physically valid (exclusive) solution and stay finite."""
    spec = replace(SPEC, degradation_cost_per_mwh=0.0, soc_initial_mwh=180.0)
    prices = np.concatenate([np.full(12, -500.0), np.full(12, 100.0)])
    lp = optimise(prices, spec, soc_end=180.0, method="lp")
    assert ((lp.charge_mw > 1e-6) & (lp.discharge_mw > 1e-6)).any()  # the loophole exists
    values = storage_values(prices, spec, soc_end=180.0)
    assert np.isfinite(values).all()


# --- water-value curve -----------------------------------------------------------------------


def test_day_type_marks_weekends_and_vic_holidays() -> None:
    import pandas as pd

    from nemspike.battery.water_value import day_type

    # Interval ends (UTC) for 10:00 AEST on: Fri 1 Nov, Sat 2 Nov, Tue 5 Nov 2024 (Melbourne Cup).
    ends = pd.Series(
        pd.to_datetime(["2024-11-01 00:05", "2024-11-02 00:05", "2024-11-05 00:05"], utc=True)
    )
    assert day_type(ends).tolist() == [0, 1, 1]


def test_curve_uses_only_history_before_cutoff() -> None:
    import pandas as pd

    from nemspike.battery.water_value import day_type, slot_of_day, water_value_curve

    ends = pd.Series(pd.date_range("2024-01-01 14:05", periods=288 * 14, freq="5min", tz="UTC"))
    values = pd.DataFrame(
        {"interval_end": ends, "value": 10.0, "slot": slot_of_day(ends), "weekend": day_type(ends)}
    )
    cutoff = ends.iloc[288 * 10]
    values.loc[values["interval_end"] > cutoff, "value"] = 9999.0  # the future
    curve = water_value_curve(values, cutoff)
    assert len(curve) == 2 * 288
    assert (curve == 10.0).all()


def test_cache_key_changes_with_battery_or_price_history() -> None:
    """Cached storage values must not be reused for a different battery or different prices."""
    import pandas as pd

    from nemspike.battery.water_value import storage_values_key

    ends = pd.Series(pd.date_range("2024-01-01 14:05", periods=288, freq="5min", tz="UTC"))
    prices = pd.DataFrame({"interval_end": ends, "rrp": 50.0})
    key = storage_values_key(prices, SPEC)
    assert key == storage_values_key(prices.copy(), SPEC)
    assert key != storage_values_key(prices, replace(SPEC, round_trip_efficiency=0.85))
    assert key != storage_values_key(prices, replace(SPEC, degradation_cost_per_mwh=10.0))
    assert key != storage_values_key(prices.iloc[:-1], SPEC)
    revised = prices.copy()
    revised.loc[100, "rrp"] = 51.0
    assert key != storage_values_key(revised, SPEC)
