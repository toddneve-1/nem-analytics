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
