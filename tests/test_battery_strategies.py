from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nemspike.battery.lp import BatterySpec
from nemspike.battery.strategies import (
    add_nem_day,
    capture_bootstrap,
    daily_results,
    naive_time_of_day,
    paired_bootstrap,
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


def test_daily_mode_returns_to_initial_state_of_charge_every_day(prices: pd.DataFrame) -> None:
    pf = perfect_foresight(prices, SPEC, "daily")
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


# --- continuous vs daily ---------------------------------------------------------------------


def test_continuous_is_at_least_as_good_as_daily(prices: pd.DataFrame) -> None:
    """Carrying energy across midnight can only add options, never remove them."""
    cont = daily_results(perfect_foresight(prices, SPEC, "continuous"), SPEC)
    daily = daily_results(perfect_foresight(prices, SPEC, "daily"), SPEC)
    assert cont["net_revenue"].sum() >= daily["net_revenue"].sum() - 1e-6


def test_continuous_only_fixes_state_of_charge_at_period_ends(prices: pd.DataFrame) -> None:
    d = perfect_foresight(prices, SPEC, "continuous")
    assert d["soc_mwh"].iloc[-1] == pytest.approx(SPEC.soc_initial_mwh)
    midnight_soc = d.groupby("nem_day")["soc_mwh"].last().iloc[:-1]
    # Not forced back to the initial level every night (it is free to carry energy over).
    assert (np.abs(midnight_soc - SPEC.soc_initial_mwh) > 1e-3).any()


def test_continuous_rejects_a_gap_in_prices(prices: pd.DataFrame) -> None:
    gappy = prices.drop(index=prices.index[3 * 288 : 4 * 288])  # remove a whole day
    with pytest.raises(ValueError, match="unbroken"):
        perfect_foresight(gappy, SPEC, "continuous")


def test_naive_continuous_schedule_carries_charge_between_days(prices: pd.DataFrame) -> None:
    train, test = prices.iloc[: 5 * 288], prices.iloc[5 * 288 :]
    d = naive_time_of_day(train, test, SPEC, "continuous")
    starts = d.groupby("nem_day")["soc_mwh"].first().to_numpy()
    ends = d.groupby("nem_day")["soc_mwh"].last().to_numpy()
    # Every day ends at the same level (a repeating cycle), so the quarter gains no net energy.
    np.testing.assert_allclose(ends, ends[0], atol=1e-9)
    assert np.isfinite(starts).all()


# --- bootstrap of differences between strategies -----------------------------------------------


def _daily(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    days = pd.date_range("2024-01-01", periods=len(a)).date
    return pd.concat(
        [
            pd.DataFrame({"strategy": name, "nem_day": days, "net_revenue": v})
            for name, v in (("a", a), ("b", b))
        ],
        ignore_index=True,
    )


def test_bootstrap_interval_excludes_zero_for_a_consistent_edge() -> None:
    rng = np.random.default_rng(0)
    b = rng.normal(1000, 300, 200)
    out = paired_bootstrap(_daily(b + rng.normal(50, 20, 200), b), "a", "b", n_boot=2000)
    assert out["ci_low"] > 0 and out["share_not_positive"] == 0.0
    assert out["ci_low"] < out["difference"] < out["ci_high"]


def test_bootstrap_interval_spans_zero_when_one_day_decides_the_total() -> None:
    """The case this exists for: ahead on almost every day, but one day carries the total."""
    a, b = np.full(200, 1100.0), np.full(200, 1000.0)
    a[57] -= 15_000.0  # one extreme day outweighs 199 small wins
    out = paired_bootstrap(_daily(a, b), "a", "b", n_boot=2000)
    assert out["difference"] == pytest.approx(199 * 100 - 14_900)
    assert out["ci_low"] < 0 < out["ci_high"]
    assert 0.05 < out["share_not_positive"] < 0.95
    # The robust views are not fooled by the one day: ahead on nearly every day, by $100.
    assert out["share_days_ahead"] == pytest.approx(199 / 200)
    assert out["share_days_ahead_ci_low"] > 0.95
    assert out["median_daily"] == out["median_daily_ci_low"] == out["median_daily_ci_high"] == 100.0


def test_capture_interval_brackets_the_observed_share() -> None:
    rng = np.random.default_rng(2)
    reference = rng.lognormal(10, 1.5, 300)  # heavy-tailed, like perfect-foresight revenue
    daily = _daily(reference * rng.uniform(0.2, 0.8, 300), reference)
    low, high = capture_bootstrap(daily, "a", reference="b", n_boot=2000)
    observed = daily[daily.strategy == "a"].net_revenue.sum() / reference.sum()
    assert 0.2 < low < observed < high < 0.8
    assert capture_bootstrap(daily, "b", reference="b", n_boot=50) == (1.0, 1.0)


def test_bootstrap_is_reproducible_and_paired() -> None:
    rng = np.random.default_rng(1)
    b = rng.normal(1000, 5000, 120)  # wild days common to both strategies
    daily = _daily(b + 10.0, b)
    one, two = (paired_bootstrap(daily, "a", "b", n_boot=500, seed=7) for _ in range(2))
    assert one == two
    # Pairing removes the shared day-to-day swings: the difference is exactly 10 a day.
    assert one["ci_low"] == pytest.approx(1200.0) and one["ci_high"] == pytest.approx(1200.0)


def test_bootstrap_requires_the_same_days_for_both_strategies() -> None:
    daily = _daily(np.ones(30), np.ones(30))
    with pytest.raises(ValueError, match="same days"):
        paired_bootstrap(daily.iloc[:-1], "a", "b", n_boot=10)
    with pytest.raises(ValueError, match="no daily results"):
        paired_bootstrap(daily, "a", "missing", n_boot=10)


def test_summary_reports_difference_from_baseline_with_interval(prices: pd.DataFrame) -> None:
    train, test = prices.iloc[: 5 * 288], prices.iloc[5 * 288 :]
    daily = daily_results(
        pd.concat([perfect_foresight(test, SPEC), naive_time_of_day(train, test, SPEC)]), SPEC
    )
    plain = summarise_strategies(daily, SPEC)
    s = summarise_strategies(daily, SPEC, baseline="naive_time_of_day", n_boot=200, block_days=2)
    pd.testing.assert_frame_equal(s[plain.columns], plain)  # existing columns are unchanged
    pf, naive = s.loc["perfect_foresight"], s.loc["naive_time_of_day"]
    assert pf["net_vs_naive_time_of_day"] == pytest.approx(pf["net_revenue"] - naive["net_revenue"])
    assert pf["net_vs_naive_time_of_day_ci_low"] <= pf["net_vs_naive_time_of_day_ci_high"]
    assert np.isnan(naive["net_vs_naive_time_of_day"])  # the baseline has no difference to itself
    assert naive["capture_vs_perfect_ci_low"] <= naive["capture_vs_perfect"]
    assert naive["capture_vs_perfect"] <= naive["capture_vs_perfect_ci_high"]
    assert pf["net_vs_naive_time_of_day_share_days_ahead"] == 1.0
