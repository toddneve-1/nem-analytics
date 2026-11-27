from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nemmodel.inputs import ModelYear
from nemmodel.network import (
    GasConstraint,
    build_network,
    daily_southern_gas_tj,
    solve,
    unit_ratings,
)
from nemmodel.outages import sample_available_fraction
from nemmodel.study import Data, eue_table, fleet
from nemmodel.traces import _norm, _to_hourly

from .conftest import unit


def test_model_year():
    y = ModelYear(2029)
    assert y.label == "2028-29"
    assert y.start == pd.Timestamp("2028-07-01")


def test_half_hours_to_hour_starting():
    ts = pd.date_range("2028-07-01 00:30", periods=4, freq="30min")  # interval ending
    df = pd.DataFrame({"datetime": ts, "value": [1.0, 3.0, 5.0, 7.0], "k": "a"})
    h = _to_hourly(df, "value", ["k"])
    assert list(h.index) == [pd.Timestamp("2028-07-01 00:00"), pd.Timestamp("2028-07-01 01:00")]
    assert h["a"].tolist() == [2.0, 6.0]


def test_name_normalisation_matches_trace_names():
    assert _norm("Cattle Hill Wind Farm") == _norm("Cattle Hill")
    assert _norm("Bungala One Solar Farm") == _norm("Bungala One")


def test_outage_share_matches_forced_outage_rate():
    units = pd.DataFrame([unit("A", "NSW1", "coal", 100, 30, for_full=0.1, mttr_full=50.0)] * 200)
    units["unit_id"] = [f"U{i}" for i in range(len(units))]
    avail = sample_available_fraction(units, 5000, np.random.default_rng(1))
    assert abs((avail == 0).to_numpy().mean() - 0.1) < 0.01


def test_partial_outage_derates():
    units = pd.DataFrame(
        [
            unit(
                "A",
                "NSW1",
                "coal",
                100,
                30,
                for_partial=0.2,
                partial_derate=0.25,
                mttr_partial=48.0,
            )
        ]
        * 200
    )
    units["unit_id"] = [f"U{i}" for i in range(len(units))]
    avail = sample_available_fraction(units, 5000, np.random.default_rng(0)).to_numpy()
    assert set(np.round(np.unique(avail), 6)) == {0.75, 1.0}
    assert abs((avail == 0.75).mean() - 0.2) < 0.01


def test_no_outage_rate_means_always_available():
    units = pd.DataFrame([unit("A", "NSW1", "wind", 100, 0)])
    avail = sample_available_fraction(units, 100, np.random.default_rng(0))
    assert (avail["A"] == 1.0).all()


def test_seasonal_ratings(cfg):
    units = pd.DataFrame(
        [
            unit(
                "A",
                "NSW1",
                "coal",
                100,
                30,
                rating_summer_typical=90,
                rating_winter_early=100,
                rating_winter_late=105,
            )
        ]
    )
    idx = pd.DatetimeIndex(["2028-07-01", "2029-01-15", "2029-06-01"])
    assert unit_ratings(units, idx, cfg)["A"].tolist() == [100, 90, 105]


def test_dispatch_and_unserved_energy(cfg, tiny):
    # NSW: 800 MW local + up to 200 MW import vs 600 MW; VIC: 300 MW local vs 400 MW demand,
    # and NSW can export 200 MW, so VIC is short by nothing; then remove the link to force a gap.
    n = build_network(tiny["units"], tiny["ics"], tiny["demand"], tiny["cf"], cfg)
    solve(n, cfg)
    p = n.generators_t.p
    assert p["VIC1 unserved"].sum() == pytest.approx(0, abs=1e-6)
    assert p["NSW1 COAL coal"].mean() == pytest.approx(500)  # cheapest unit runs flat out
    ics = tiny["ics"].assign(fwd_summer=0, fwd_winter=0, rev_summer=0, rev_winter=50)
    n = build_network(tiny["units"], ics, tiny["demand"], tiny["cf"], cfg)
    solve(n, cfg)
    assert n.generators_t.p["VIC1 unserved"].mean() == pytest.approx(50)


def test_outages_reduce_availability(cfg, tiny):
    avail = pd.DataFrame(1.0, index=range(48), columns=["COAL", "GASN", "GASV"])
    avail["COAL"] = 0.0
    n = build_network(tiny["units"], tiny["ics"], tiny["demand"], tiny["cf"], cfg, avail)
    solve(n, cfg)
    # 600 MW of gas left against 1,000 MW of demand (how the shortfall splits is arbitrary).
    unserved = n.generators_t.p[["NSW1 unserved", "VIC1 unserved"]].sum(axis=1)
    assert unserved.mean() == pytest.approx(400)


def test_gas_cap_binds(cfg, tiny):
    units = tiny["units"][tiny["units"]["group"] != "coal"]
    n = build_network(units, tiny["ics"], tiny["demand"], tiny["cf"], cfg)
    solve(n, cfg, GasConstraint(["NSW1", "VIC1"], [7], cap_tj_per_day=72.0))
    burn = daily_southern_gas_tj(n, ["NSW1", "VIC1"])
    # 72 TJ/day at 10 GJ/MWh = 7,200 MWh/day = 300 MW on average: 700 MW short.
    assert burn.max() == pytest.approx(72.0, rel=1e-6)
    unserved = n.generators_t.p[["NSW1 unserved", "VIC1 unserved"]].sum(axis=1)
    assert unserved.mean() == pytest.approx(700, rel=1e-6)


def test_early_coal_case(cfg, committed_units):
    early = fleet(committed_units, "early", cfg)
    gone = set(committed_units["station"]) - set(early["station"])
    assert gone == {"Gladstone", "Callide B", "Bayswater", "Vales Point B"}


def test_committed_inputs(committed_units):
    u = committed_units
    assert u["group"].notna().all()
    assert not u["station"].isin(["Eraring", "Yallourn W"]).any()  # closed before FY2028-29
    thermal = u[u["group"].isin(["coal", "ccgt", "ocgt", "recip", "liquid"])]
    assert (thermal["srmc"] > 0).all()
    assert (u.loc[u["group"].isin(["battery", "pumped_hydro"]), "storage_mwh"] > 0).all()


def test_eue_weighting(cfg):
    idx = pd.date_range("2028-07-01", periods=10, freq="h")
    regions = ["QLD1", "NSW1", "VIC1", "SA1", "TAS1"]
    demand = pd.DataFrame(
        100.0, index=idx, columns=[f"{r}_{p}" for r in regions for p in ("POE10", "POE50")]
    )
    data = Data(pd.DataFrame(), pd.DataFrame(), demand, pd.DataFrame())
    rows = [
        {"case": "c", "poe": p, "sample": s, **{f"unserved_mwh_{r}": 0.0 for r in regions}}
        for p in ("POE10", "POE50")
        for s in range(2)
    ]
    rows[0]["unserved_mwh_SA1"] = 2.0  # POE10, one of two samples
    t = eue_table(pd.DataFrame(rows), data, cfg).set_index("region")
    assert t.loc["SA1", "eue_mwh"] == pytest.approx(0.304 * 1.0)
    assert t.loc["SA1", "eue_pct"] == pytest.approx(100 * 0.304 / 1000)
    assert not t.loc["SA1", "meets_standard"]
    assert t.loc["NSW1", "meets_standard"]


def test_monthly_inflow_spread_evenly():
    from nemmodel.network import monthly_inflow_mw

    t = pd.DataFrame({"scheme": "s", "reference_year": 2018, "month": ["Jul", "Aug"],
                      "inflow": [744.0, 0.0]})  # fmt: off
    idx = pd.date_range("2028-07-01", "2028-08-31 23:00", freq="h")
    mw = monthly_inflow_mw(t, "s", 2018, idx)
    assert mw.loc["2028-07"].eq(1000.0).all()  # 744 GWh over 744 hours
    assert mw.loc["2028-08"].eq(0.0).all()


def test_seasonal_hydro_carries_water_between_windows(cfg, tiny):
    """Water arriving on day 1 is needed on day 3. Weekly-style windows (here 24 h) cannot move
    it; the whole-period plan can."""
    import copy

    cfg = copy.deepcopy(cfg)
    cfg["dispatch"]["window_hours"] = 24
    cfg["hydro"]["tasmania"].update(storage_gwh=10.0, start_level=0.0, min_level=0.0)
    idx = pd.date_range("2028-07-01", periods=72, freq="h")
    units = pd.DataFrame([unit("H1", "TAS1", "hydro", 200, 0)])
    demand = pd.DataFrame(0.0, index=idx, columns=["QLD1", "NSW1", "VIC1", "SA1", "TAS1"])
    demand.loc[idx[48:], "TAS1"] = 100.0  # 2,400 MWh needed on day 3
    inflows = pd.DataFrame({"scheme": "tasmania", "reference_year": cfg["trace_reference_year"],
                            "month": ["Jul"], "inflow": [0.0]})  # fmt: off
    n = build_network(units, tiny["ics"].iloc[0:0], demand, pd.DataFrame(index=idx), cfg,
                      hydro_inflows=inflows)  # fmt: off
    # Replace the monthly inflow with 2,400 MWh arriving in the first 24 hours.
    n.storage_units_t.inflow = pd.DataFrame(
        {"TAS1 tasmania hydro": np.where(np.arange(72) < 24, 100.0, 0.0)}, index=idx
    )
    solve(n, cfg)
    assert n.generators_t.p["TAS1 unserved"].sum() == pytest.approx(0, abs=1e-3)
    soc = n.storage_units_t.state_of_charge["TAS1 tasmania hydro"]
    plan = n.seasonal_plan["TAS1 tasmania hydro"]
    assert soc.iloc[[23, 47, 71]].to_numpy() == pytest.approx(plan.iloc[[23, 47, 71]].to_numpy())


def test_seasonal_hydro_respects_floor_and_returns_to_start(cfg, tiny):
    import copy

    cfg = copy.deepcopy(cfg)
    cfg["dispatch"]["window_hours"] = 24
    cfg["hydro"]["tasmania"].update(storage_gwh=10.0, start_level=0.5, min_level=0.3)
    idx = pd.date_range("2028-07-01", periods=72, freq="h")
    units = pd.DataFrame([unit("H1", "TAS1", "hydro", 500, 0)])
    demand = pd.DataFrame(0.0, index=idx, columns=["QLD1", "NSW1", "VIC1", "SA1", "TAS1"])
    demand["TAS1"] = 200.0  # 14.4 GWh wanted; only 2 GWh may be drawn (5 GWh start, 3 floor)
    inflows = pd.DataFrame({"scheme": "tasmania", "reference_year": cfg["trace_reference_year"],
                            "month": ["Jul"], "inflow": [0.0]})  # fmt: off
    n = build_network(units, tiny["ics"].iloc[0:0], demand, pd.DataFrame(index=idx), cfg,
                      hydro_inflows=inflows)  # fmt: off
    solve(n, cfg)
    soc = n.storage_units_t.state_of_charge["TAS1 tasmania hydro"]
    assert soc.min() >= 3000 - 1e-3
    assert soc.iloc[-1] == pytest.approx(5000)  # ends where it started, so nothing is drawn
    assert n.generators_t.p["TAS1 unserved"].sum() == pytest.approx(14400)


@pytest.mark.parametrize(
    ("mode", "expected_unserved"), [("exact_units", 600.0), ("linearised", 0.0)]
)
def test_minimum_stable_level_and_down_time(cfg, tiny, mode, expected_unserved):
    """Coal (500 MW, minimum 250 MW, 8 h minimum off time) and 300 MW of gas serve 400 MW,
    except a 100 MW dip in hours 10-11. On/off decisions per unit force coal off through the dip
    and keep it off until hour 18, leaving 100 MW short in hours 12-17. The linear stand-in can
    run the station "40 % online" through the dip, so nothing is short: its known weakness."""
    import copy

    cfg = copy.deepcopy(cfg)
    cfg["dispatch"]["window_hours"] = 24
    cfg["unit_commitment"].update(mode=mode, min_up_hours={"coal": 2, "ccgt": 2},
                                  min_down_hours={"coal": 8, "ccgt": 2})  # fmt: off
    idx = pd.date_range("2028-07-01", periods=24, freq="h")
    units = pd.DataFrame([unit("COAL", "NSW1", "coal", 500, 30, msl_mw=250.0),
                          unit("GAS", "NSW1", "ocgt", 300, 150)])  # fmt: off
    demand = pd.DataFrame(0.0, index=idx, columns=["QLD1", "NSW1", "VIC1", "SA1", "TAS1"])
    demand["NSW1"] = 400.0
    demand.loc[idx[10:12], "NSW1"] = 100.0
    n = build_network(units, tiny["ics"].iloc[0:0], demand, pd.DataFrame(index=idx), cfg)
    solve(n, cfg)
    assert n.generators_t.p["NSW1 unserved"].sum() == pytest.approx(expected_unserved, abs=1e-3)


def test_ramp_limit(cfg, tiny):
    import copy

    cfg = copy.deepcopy(cfg)
    cfg["dispatch"]["window_hours"] = 24
    cfg["unit_commitment"]["enabled"] = False
    idx = pd.date_range("2028-07-01", periods=24, freq="h")
    # 500 MW coal that can move 2.5 MW/min = 150 MW/h; gas fills what coal cannot reach.
    coal = unit("COAL", "NSW1", "coal", 500, 30, ramp_up_mw_min=2.5, ramp_down_mw_min=2.5)
    units = pd.DataFrame([coal, unit("GAS", "NSW1", "ocgt", 500, 150)])
    demand = pd.DataFrame(0.0, index=idx, columns=["QLD1", "NSW1", "VIC1", "SA1", "TAS1"])
    demand["NSW1"] = 100.0
    demand.loc[idx[12:], "NSW1"] = 500.0
    n = build_network(units, tiny["ics"].iloc[0:0], demand, pd.DataFrame(index=idx), cfg)
    solve(n, cfg)
    coal = n.generators_t.p["NSW1 COAL coal"]
    assert coal.diff().abs().max() <= 150 + 1e-6


def test_gas_balance_refuses_missing_gsoo_values(tmp_path):
    from nemmodel.gas import MissingGasDataError, load_balance

    from .conftest import ROOT

    with pytest.raises(MissingGasDataError, match="source"):
        load_balance(ROOT / "config" / "gas_balance.yaml")  # committed template is blank


def test_gas_balance_daily_allowance():
    from nemmodel.gas import daily_allowance

    w = {"southern_production_tj_d": 1000.0, "northern_supply_tj_d": 300.0,
         "other_demand_peak_tj_d": 1500.0, "other_demand_average_tj_d": 1100.0}  # fmt: off
    balance = {"winters": {2028: w}, "storage_withdrawal_tj_d": 500.0}
    idx = pd.date_range("2028-07-01", periods=3 * 24, freq="h")
    demand = pd.Series(np.repeat([100.0, 200.0, 300.0], 24), index=idx)  # mean 200, max 300
    a = daily_allowance(balance, pd.DataFrame(), demand, [7])
    supply = 1000 + 300 + 500
    # Day 1 below average: average demand. Day 2 average. Day 3 the coldest: peak demand.
    assert a.tolist() == pytest.approx([supply - 1100, supply - 1100, supply - 1500])


def test_gas_cap_varies_by_day(cfg, tiny):
    units = tiny["units"][tiny["units"]["group"] != "coal"]
    n = build_network(units, tiny["ics"], tiny["demand"], tiny["cf"], cfg)
    days = pd.DatetimeIndex(["2028-07-01", "2028-07-02"])
    solve(n, cfg, GasConstraint(["NSW1", "VIC1"], [7], pd.Series([72.0, 24.0], index=days)))
    burn = daily_southern_gas_tj(n, ["NSW1", "VIC1"])
    assert burn.tolist() == pytest.approx([72.0, 24.0], rel=1e-6)
