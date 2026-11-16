from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nemspike.battery.lp import BatterySpec
from nemspike.battery.strategies import (
    add_nem_day,
    daily_results,
    naive_time_of_day,
    perfect_foresight,
    summarise_strategies,
    time_of_day_schedule,
)
from tests.conftest import make_prices

SPEC = BatterySpec(100.0, 200.0, 0.9, 20.0, 180.0, 100.0)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return make_prices(days=10, seed=11)


def test_nem_day_and_slot_use_interval_start_in_aest(prices: pd.DataFrame) -> None:
    p = add_nem_day(prices)
    first = p.iloc[0]  # interval ending 00:05 AEST on 2 Jan 2024 (synthetic data starts there)
    assert first["slot"] == 0
    last_of_day = p[p["nem_day"] == first["nem_day"]].iloc[-1]
    assert last_of_day["slot"] == 287  # the interval ending 24:00 belongs to the same day
    assert p.groupby("nem_day").size().eq(288).all()


def test_perfect_foresight_beats_naive_on_every_day(prices: pd.DataFrame) -> None:
    train, test = prices.iloc[: 5 * 288], prices.iloc[5 * 288 :]
    daily = daily_results(
        pd.concat([perfect_foresight(test, SPEC), naive_time_of_day(train, test, SPEC)]), SPEC
    )
    wide = daily.pivot_table(index="nem_day", columns="strategy", values="net_revenue")
    assert (wide["perfect_foresight"] >= wide["naive_time_of_day"] - 1e-6).all()


def test_each_day_returns_to_initial_state_of_charge(prices: pd.DataFrame) -> None:
    pf = perfect_foresight(prices, SPEC)
    end_soc = pf.groupby("nem_day")["soc_mwh"].last()
    np.testing.assert_allclose(end_soc, SPEC.soc_initial_mwh, atol=1e-6)


def test_naive_schedule_depends_only_on_training_prices(prices: pd.DataFrame) -> None:
    train, test = prices.iloc[: 5 * 288], prices.iloc[5 * 288 :]
    a = naive_time_of_day(train, test, SPEC)
    scrambled = test.assign(rrp=np.random.default_rng(0).permutation(test["rrp"].to_numpy()))
    b = naive_time_of_day(train, scrambled, SPEC)
    # Same dispatch whatever happens in the test period; only realised revenue differs.
    np.testing.assert_array_equal(a["charge_mw"], b["charge_mw"])
    np.testing.assert_array_equal(a["discharge_mw"], b["discharge_mw"])


def test_time_of_day_schedule_charges_in_cheap_slots(prices: pd.DataFrame) -> None:
    d = time_of_day_schedule(prices, SPEC)
    profile = add_nem_day(prices).groupby("slot")["rrp"].mean().to_numpy()
    assert profile[d.charge_mw > 1].mean() < profile[d.discharge_mw > 1].mean()


def test_summary_capture_is_relative_to_perfect_foresight(prices: pd.DataFrame) -> None:
    train, test = prices.iloc[: 5 * 288], prices.iloc[5 * 288 :]
    daily = daily_results(
        pd.concat([perfect_foresight(test, SPEC), naive_time_of_day(train, test, SPEC)]), SPEC
    )
    s = summarise_strategies(daily, SPEC)
    assert s.loc["perfect_foresight", "capture_vs_perfect"] == pytest.approx(1.0)
    # The naive schedule can lose money (it can chase spikes that do not recur), so only the
    # upper bound is guaranteed.
    assert s.loc["naive_time_of_day", "capture_vs_perfect"] <= 1.0
