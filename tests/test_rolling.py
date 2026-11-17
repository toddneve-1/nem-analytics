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
    return pd.DataFrame(cols, index=origins)


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
