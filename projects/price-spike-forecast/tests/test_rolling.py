from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nemspike.battery.lp import BatterySpec, optimise
from nemspike.battery.rolling import DEFAULT_HORIZONS, plan_price_path, rolling_dispatch
from nemspike.battery.strategies import daily_results, perfect_foresight
from nemspike.timeutils import INTERVAL
from tests.conftest import make_prices

SPEC = BatterySpec(100.0, 200.0, 0.9, 20.0, 180.0, 100.0, degradation_cost_per_mwh=40.0)


def flat_curve(value: float) -> pd.Series:
    idx = pd.MultiIndex.from_product([[0, 1], range(288)], names=["weekend", "slot"])
    return pd.Series(value, index=idx)


FIVE_MIN = tuple(range(5, 245, 5))


def oracle_plan(prices: pd.DataFrame, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> pd.DataFrame:
    """Plan prices equal to the actual price at each horizon's interval (a perfect forecast)."""
    s = prices.set_index("interval_end")["rrp"]
    origins = s.index - INTERVAL
    cols = {h: s.reindex(origins + pd.Timedelta(minutes=h)).to_numpy() for h in horizons}
    # Past the end of the synthetic series there is no "actual" to copy: carry the last known
    # price forward (real development folds have data beyond them, so this never arises there).
    return pd.DataFrame(cols, index=origins).ffill(axis=1).fillna(float(s.iloc[-1]))


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return make_prices(days=4, seed=21)


def test_plan_price_path_uses_30_minute_blocks() -> None:
    row = pd.Series({h: float(h) for h in DEFAULT_HORIZONS})
    path = plan_price_path(row, 12, DEFAULT_HORIZONS)
    assert path.tolist() == [30.0] * 6 + [60.0] * 6


def test_terminal_value_controls_holding_versus_selling() -> None:
    prices = np.full(12, 100.0)
    sell = optimise(prices, SPEC, soc_start=150.0, terminal_value=0.0)
    hold = optimise(prices, SPEC, soc_start=150.0, terminal_value=200.0)
    assert sell.soc_mwh[-1] < 150.0  # leftover energy worthless: sell some now
    assert hold.soc_mwh[-1] > 150.0  # leftover energy valuable: charge and keep it


def test_rolling_dispatch_is_physically_valid_and_ends_at_initial_level(prices) -> None:
    d = rolling_dispatch(prices, oracle_plan(prices), flat_curve(60.0), SPEC, "forecast")
    eta, dt = SPEC.eta_one_way, 5 / 60
    prev = np.concatenate([[SPEC.soc_initial_mwh], d["soc_mwh"].to_numpy()[:-1]])
    np.testing.assert_allclose(
        d["soc_mwh"], prev + eta * d["charge_mw"] * dt - d["discharge_mw"] * dt / eta, atol=1e-6
    )
    assert d["soc_mwh"].min() >= 20 - 1e-6 and d["soc_mwh"].max() <= 180 + 1e-6
    assert d["soc_mwh"].iloc[-1] == pytest.approx(SPEC.soc_initial_mwh)
    assert not ((d["charge_mw"] > 1e-6) & (d["discharge_mw"] > 1e-6)).any()


def test_rolling_with_perfect_5_minute_forecast_approaches_perfect_foresight(prices):
    """With a perfect forecast at 5-minute resolution, rolling 4-hour planning should capture most
    of the perfect-foresight value (it only lacks sight beyond 4 hours), and never exceed it."""
    plan = oracle_plan(prices, FIVE_MIN)
    roll = daily_results(
        rolling_dispatch(prices, plan, flat_curve(60.0), SPEC, "forecast", horizons=FIVE_MIN),
        SPEC,
    )
    pf = daily_results(perfect_foresight(prices, SPEC, "continuous"), SPEC)
    roll_net, pf_net = roll["net_revenue"].sum(), pf["net_revenue"].sum()
    assert roll_net <= pf_net + 1e-6
    assert roll_net > 0.8 * pf_net


def test_30_minute_grid_loses_5_minute_detail(prices):
    """Even a perfect forecast on a 30-minute grid cannot place single-interval spikes, so it
    earns less than the same forecast at 5-minute resolution."""
    fine = rolling_dispatch(
        prices, oracle_plan(prices, FIVE_MIN), flat_curve(60.0), SPEC, "f", horizons=FIVE_MIN
    )
    coarse = rolling_dispatch(prices, oracle_plan(prices), flat_curve(60.0), SPEC, "c")
    assert (
        daily_results(coarse, SPEC)["net_revenue"].sum()
        < daily_results(fine, SPEC)["net_revenue"].sum()
    )


def test_decisions_do_not_depend_on_later_actual_prices(prices) -> None:
    """Leakage check: scramble actual prices after a cut-off; dispatch before it must not change,
    because decisions use only forecasts available at each origin."""
    plan = oracle_plan(prices)  # forecasts fixed in advance
    base = rolling_dispatch(prices, plan, flat_curve(60.0), SPEC, "forecast")
    cut = prices["interval_end"].iloc[2 * 288]
    scrambled = prices.copy()
    later = scrambled["interval_end"] > cut
    scrambled.loc[later, "rrp"] = np.random.default_rng(0).normal(5000, 3000, later.sum())
    alt = rolling_dispatch(scrambled, plan, flat_curve(60.0), SPEC, "forecast")
    before = base["interval_end"] <= cut
    np.testing.assert_array_equal(base.loc[before, "charge_mw"], alt.loc[before, "charge_mw"])
    np.testing.assert_array_equal(base.loc[before, "discharge_mw"], alt.loc[before, "discharge_mw"])


def test_missing_forecast_origin_raises(prices) -> None:
    plan = oracle_plan(prices).iloc[10:]
    with pytest.raises(KeyError):
        rolling_dispatch(prices, plan, flat_curve(60.0), SPEC, "forecast")


# --- threshold execution ---------------------------------------------------------------------

from nemspike.battery.rolling import execution_thresholds, threshold_dispatch  # noqa: E402


def climatology(prices: pd.DataFrame) -> pd.Series:
    from nemspike.battery.strategies import add_nem_day

    return add_nem_day(prices).groupby("slot")["rrp"].mean()


def _check_valid(d: pd.DataFrame) -> None:
    eta, dt = SPEC.eta_one_way, 5 / 60
    prev = np.concatenate([[SPEC.soc_initial_mwh], d["soc_mwh"].to_numpy()[:-1]])
    np.testing.assert_allclose(
        d["soc_mwh"], prev + eta * d["charge_mw"] * dt - d["discharge_mw"] * dt / eta, atol=1e-6
    )
    assert d["soc_mwh"].min() >= 20 - 1e-6 and d["soc_mwh"].max() <= 180 + 1e-6
    assert d["charge_mw"].max() <= 100 + 1e-6 and d["discharge_mw"].max() <= 100 + 1e-6
    assert not ((d["charge_mw"] > 1e-6) & (d["discharge_mw"] > 1e-6)).any()
    assert d["soc_mwh"].iloc[-1] == pytest.approx(SPEC.soc_initial_mwh)


def test_execution_thresholds_account_for_losses_and_wear() -> None:
    below, above = execution_thresholds(100.0, SPEC)
    eta = SPEC.eta_one_way
    assert below == pytest.approx(100 * eta)  # buying is worth it below the stored value x eta
    assert above == pytest.approx(100 / eta + 40.0)  # selling must beat value / eta plus wear


@pytest.mark.parametrize("use_forecast", [True, False])
def test_threshold_dispatch_is_physically_valid(prices, use_forecast: bool) -> None:
    plan = oracle_plan(prices) if use_forecast else None
    d = threshold_dispatch(
        prices, plan, flat_curve(60.0), SPEC, "t", climatology=climatology(prices)
    )
    _check_valid(d)


@pytest.mark.parametrize("use_forecast", [True, False])
def test_threshold_decisions_do_not_depend_on_later_prices(prices, use_forecast: bool) -> None:
    """Each interval may react to its own realised price (a price-contingent offer) but never to
    later prices."""
    plan = oracle_plan(prices) if use_forecast else None
    clim = climatology(prices)
    base = threshold_dispatch(prices, plan, flat_curve(60.0), SPEC, "t", climatology=clim)
    cut = prices["interval_end"].iloc[2 * 288]
    scrambled = prices.copy()
    later = scrambled["interval_end"] > cut
    scrambled.loc[later, "rrp"] = np.random.default_rng(1).normal(5000, 3000, later.sum())
    alt = threshold_dispatch(scrambled, plan, flat_curve(60.0), SPEC, "t", climatology=clim)
    before = base["interval_end"] <= cut
    np.testing.assert_array_equal(base.loc[before, "charge_mw"], alt.loc[before, "charge_mw"])
    np.testing.assert_array_equal(base.loc[before, "discharge_mw"], alt.loc[before, "discharge_mw"])


def test_threshold_only_reacts_to_price_against_historical_value(prices) -> None:
    """With a historical value of $60/MWh stored: charge below 60 x eta, discharge above
    60 / eta + wear, idle in between (away from the end-of-period approach)."""
    d = threshold_dispatch(
        prices, None, flat_curve(60.0), SPEC, "t", climatology=climatology(prices)
    ).iloc[:-48]
    below, above = execution_thresholds(60.0, SPEC)
    charging = d["charge_mw"] > 1e-6
    discharging = d["discharge_mw"] > 1e-6
    assert (d.loc[charging, "rrp"] < below).all()
    assert (d.loc[discharging, "rrp"] > above).all()


# --- offer-curve execution (option B) ---------------------------------------------------------


def test_offer_curve_is_physically_valid(prices) -> None:
    d = threshold_dispatch(prices, oracle_plan(prices), flat_curve(60.0), SPEC, "b", offer="curve")
    _check_valid(d)


def test_offer_curve_decisions_do_not_depend_on_later_prices(prices) -> None:
    plan = oracle_plan(prices)
    base = threshold_dispatch(prices, plan, flat_curve(60.0), SPEC, "b", offer="curve")
    cut = prices["interval_end"].iloc[2 * 288]
    scrambled = prices.copy()
    later = scrambled["interval_end"] > cut
    scrambled.loc[later, "rrp"] = np.random.default_rng(2).normal(5000, 3000, later.sum())
    alt = threshold_dispatch(scrambled, plan, flat_curve(60.0), SPEC, "b", offer="curve")
    before = base["interval_end"] <= cut
    np.testing.assert_array_equal(base.loc[before, "charge_mw"], alt.loc[before, "charge_mw"])
    np.testing.assert_array_equal(base.loc[before, "discharge_mw"], alt.loc[before, "discharge_mw"])


def test_offer_curve_gives_partial_quantities_unlike_full_power_thresholds(prices) -> None:
    """The offer curve can trade less than full power when the plan's value of stored energy
    changes with the charge level; the single-threshold rule only ever trades at full power or
    up to a limit."""
    plan = oracle_plan(prices)
    curve_d = threshold_dispatch(prices, plan, flat_curve(60.0), SPEC, "b", offer="curve")
    body = curve_d.iloc[:-48]
    traded = (body["charge_mw"] + body["discharge_mw"]) > 1e-6
    partial = traded & ((body["charge_mw"] + body["discharge_mw"]) < 100 - 1e-6)
    assert partial.any()


def test_offer_curve_requires_forecasts(prices) -> None:
    with pytest.raises(ValueError):
        threshold_dispatch(
            prices,
            None,
            flat_curve(60.0),
            SPEC,
            "b",
            climatology=climatology(prices),
            offer="curve",
        )
