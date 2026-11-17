from __future__ import annotations

import itertools

import numpy as np
import pytest

from nemspike.battery.dp import optimise_dp
from nemspike.battery.lp import DT_HOURS, BatterySpec, optimise

SPEC = BatterySpec(
    power_mw=100.0,
    energy_mwh=200.0,
    round_trip_efficiency=0.9,
    soc_min_mwh=20.0,
    soc_max_mwh=180.0,
    soc_initial_mwh=100.0,
)
TOL = 1e-6


def _check_physics(d, spec: BatterySpec, soc_start: float) -> None:
    eta = spec.eta_one_way
    soc_prev = np.concatenate([[soc_start], d.soc_mwh[:-1]])
    expected = soc_prev + eta * d.charge_mw * DT_HOURS - d.discharge_mw * DT_HOURS / eta
    np.testing.assert_allclose(d.soc_mwh, expected, atol=1e-6)
    assert (d.charge_mw <= spec.power_mw + TOL).all() and (
        d.discharge_mw <= spec.power_mw + TOL
    ).all()
    assert (d.soc_mwh >= spec.soc_min_mwh - TOL).all() and (
        d.soc_mwh <= spec.soc_max_mwh + TOL
    ).all()


def test_buys_low_sells_high() -> None:
    prices = np.array([10.0] * 12 + [200.0] * 12)
    d = optimise(prices, SPEC, soc_end=SPEC.soc_initial_mwh)
    _check_physics(d, SPEC, SPEC.soc_initial_mwh)
    assert d.charge_mw[:12].sum() > 0 and d.discharge_mw[12:].sum() > 0
    assert d.charge_mw[12:].sum() == pytest.approx(0, abs=TOL)
    assert d.revenue(prices) > 0


def test_revenue_matches_hand_calculation() -> None:
    """One hour at $0 then one hour at $100, starting empty, ending empty."""
    spec = BatterySpec(100.0, 200.0, 0.81, 0.0, 200.0, 0.0)  # eta one-way = 0.9
    prices = np.array([0.0] * 12 + [100.0] * 12)
    d = optimise(prices, spec, soc_end=0.0)
    # Charge 100 MW for 1 h -> stores 90 MWh; discharge 81 MWh delivered at $100 = $8,100.
    assert d.revenue(prices) == pytest.approx(8100.0, rel=1e-6)


def test_no_trade_when_spread_does_not_cover_losses() -> None:
    prices = np.array([100.0] * 12 + [105.0] * 12)  # 105 * 0.9 < 100
    d = optimise(prices, SPEC, soc_end=SPEC.soc_initial_mwh)
    assert d.throughput_mwh() == pytest.approx(0, abs=1e-6)


def test_terminal_state_of_charge_is_enforced() -> None:
    prices = np.full(24, 500.0)  # tempting to sell everything...
    d = optimise(prices, SPEC, soc_end=SPEC.soc_initial_mwh)
    assert d.soc_mwh[-1] == pytest.approx(SPEC.soc_initial_mwh)
    free = optimise(prices, SPEC, soc_end=None)
    assert free.soc_mwh[-1] == pytest.approx(SPEC.soc_min_mwh)  # ...which it does when free


def _overlaps(d) -> bool:
    return bool(((d.charge_mw > 1e-6) & (d.discharge_mw > 1e-6)).any())


def test_lp_relaxation_burns_energy_at_negative_prices_but_exact_methods_do_not() -> None:
    """At negative prices with a full battery, the pure LP charges and discharges at once to be
    paid for consuming energy through losses. The MILP and the hybrid must not."""
    spec = BatterySpec(100.0, 200.0, 0.9, 20.0, 180.0, 180.0)  # starts full
    prices = np.full(12, -500.0)
    lp = optimise(prices, spec, soc_end=180.0, method="lp")
    assert _overlaps(lp)
    for method in ("milp", "hybrid"):
        d = optimise(prices, spec, soc_end=180.0, method=method)
        assert not _overlaps(d)
        _check_physics(d, spec, 180.0)
        assert lp.revenue(prices) > d.revenue(prices)


@pytest.mark.parametrize("seed", range(5))
def test_hybrid_matches_exact_milp_on_mixed_sign_prices(seed: int) -> None:
    """Days with negative middays force the dynamic-programme fallback; it should be within a
    small fraction of the exact MILP and never above the LP upper bound."""
    rng = np.random.default_rng(seed)
    hours = np.arange(48) / 2  # 4 hours at 5-minute resolution, compressed "day"
    prices = 80 * np.sin(hours / 24 * 2 * np.pi * 6) + rng.normal(0, 30, 48)
    prices[10:22] -= 150  # a negative stretch
    lp = optimise(prices, SPEC, soc_end=SPEC.soc_initial_mwh, method="lp").revenue(prices)
    exact = optimise(prices, SPEC, soc_end=SPEC.soc_initial_mwh, method="milp").revenue(prices)
    hybrid_d = optimise(prices, SPEC, soc_end=SPEC.soc_initial_mwh, method="hybrid")
    hybrid = hybrid_d.revenue(prices)
    assert not _overlaps(hybrid_d)
    _check_physics(hybrid_d, SPEC, SPEC.soc_initial_mwh)
    assert hybrid <= exact + 1e-6 <= lp + 2e-6
    assert hybrid >= exact - 0.01 * abs(exact) - 1.0


def test_dynamic_programme_hits_full_power_exactly() -> None:
    """The grid is chosen so full charge and discharge power are representable (RTE 0.9)."""
    prices = np.array([-100.0] * 6 + [500.0] * 6)
    d = optimise_dp(prices, SPEC, soc_end=None)
    assert d.charge_mw.max() == pytest.approx(SPEC.power_mw)
    assert d.discharge_mw.max() == pytest.approx(SPEC.power_mw)
    _check_physics(d, SPEC, SPEC.soc_initial_mwh)


def test_random_prices_respect_all_constraints() -> None:
    rng = np.random.default_rng(0)
    prices = np.concatenate([rng.normal(60, 50, 288 - 10), rng.uniform(300, 5000, 10)])
    rng.shuffle(prices)
    d = optimise(prices, SPEC, soc_end=SPEC.soc_initial_mwh)
    _check_physics(d, SPEC, SPEC.soc_initial_mwh)
    assert not _overlaps(d)
    # Optimal revenue is at least that of doing nothing.
    assert d.revenue(prices) >= -1e-6


def test_rejects_nan_prices() -> None:
    with pytest.raises(ValueError):
        optimise(np.array([1.0, np.nan]), SPEC)


# --- degradation cost ------------------------------------------------------------------------


def _with_cost(spec: BatterySpec, k: float) -> BatterySpec:
    from dataclasses import replace

    return replace(spec, degradation_cost_per_mwh=k)


def test_degradation_cost_hand_calculation() -> None:
    """$0 then $100 for an hour each, starting empty: 81 MWh delivered, wear $10/MWh -> $810."""
    spec = _with_cost(BatterySpec(100.0, 200.0, 0.81, 0.0, 200.0, 0.0), 10.0)
    prices = np.array([0.0] * 12 + [100.0] * 12)
    d = optimise(prices, spec, soc_end=0.0)
    assert d.revenue(prices) == pytest.approx(8100.0, rel=1e-6)
    assert d.degradation_cost(spec) == pytest.approx(810.0, rel=1e-6)


def test_no_trade_when_spread_does_not_cover_losses_plus_wear() -> None:
    # 120 * 0.9 = 108 covers the $100 purchase on losses alone, but not with $10/MWh wear:
    # per MWh bought, revenue 0.9 * 120 - 100 = 8 < wear 0.9 * 10 = 9.
    prices = np.array([100.0] * 12 + [120.0] * 12)
    free = optimise(prices, SPEC, soc_end=SPEC.soc_initial_mwh)
    assert free.throughput_mwh() > 1.0
    worn = optimise(prices, _with_cost(SPEC, 10.0), soc_end=SPEC.soc_initial_mwh)
    assert worn.throughput_mwh() == pytest.approx(0, abs=1e-6)


def test_higher_degradation_cost_never_increases_throughput() -> None:
    rng = np.random.default_rng(3)
    prices = rng.normal(60, 60, 288)
    throughput = [
        optimise(prices, _with_cost(SPEC, k), soc_end=SPEC.soc_initial_mwh).throughput_mwh()
        for k in (0.0, 10.0, 20.0, 50.0)
    ]
    assert all(a >= b - 1e-6 for a, b in itertools.pairwise(throughput))
    assert throughput[0] > throughput[-1]


@pytest.mark.parametrize("seed", range(3))
def test_hybrid_matches_milp_with_degradation_cost(seed: int) -> None:
    rng = np.random.default_rng(seed)
    prices = 80 * np.sin(np.arange(48) / 8) + rng.normal(0, 30, 48)
    prices[10:22] -= 150
    spec = _with_cost(SPEC, 10.0)

    def net(method: str) -> float:
        d = optimise(prices, spec, soc_end=SPEC.soc_initial_mwh, method=method)
        return d.revenue(prices) - d.degradation_cost(spec)

    exact, hybrid = net("milp"), net("hybrid")
    assert hybrid <= exact + 1e-6
    assert hybrid >= exact - 0.01 * abs(exact) - 1.0


# --- cyclic (repeating) schedules ------------------------------------------------------------


def _typical_day() -> np.ndarray:
    """Shape like the real VIC1 average day: morning shoulder, solar trough, evening peak, night."""
    hours = np.arange(288) / 12
    p = np.full(288, 81.0)  # overnight
    p[(hours >= 6) & (hours < 7)] = 88.0  # morning shoulder
    p[(hours >= 11) & (hours < 14)] = 1.0  # midday solar trough
    p[(hours >= 17) & (hours < 19)] = 180.0  # evening peak
    return p


def _net(d, prices: np.ndarray, spec: BatterySpec) -> float:
    return d.revenue(prices) - d.degradation_cost(spec)


def test_cyclic_schedule_wraps_around_and_respects_limits() -> None:
    spec = _with_cost(SPEC, 40.0)
    prices = _typical_day()
    d = optimise(prices, spec, cyclic=True)
    eta = spec.eta_one_way
    soc_prev = np.concatenate([[d.soc_mwh[-1]], d.soc_mwh[:-1]])  # wrap: day starts where it ended
    expected = soc_prev + eta * d.charge_mw * DT_HOURS - d.discharge_mw * DT_HOURS / eta
    np.testing.assert_allclose(d.soc_mwh, expected, atol=1e-6)
    assert d.soc_mwh.min() >= spec.soc_min_mwh - TOL and d.soc_mwh.max() <= spec.soc_max_mwh + TOL


def test_cyclic_schedule_is_never_worse_and_skips_forced_night_refill() -> None:
    spec = _with_cost(SPEC, 40.0)
    prices = _typical_day()
    forced = optimise(prices, spec, soc_start=100.0, soc_end=100.0)
    cyclic = optimise(prices, spec, cyclic=True)
    assert _net(cyclic, prices, spec) >= _net(forced, prices, spec) - 1e-6
    night = (np.arange(288) / 12) >= 19
    # Buying at $81 to sell at $88 loses money after losses and wear; the free schedule avoids it.
    assert cyclic.charge_mw[night].sum() == pytest.approx(0, abs=1e-6)
    assert _net(cyclic, prices, spec) > _net(forced, prices, spec)


def test_cyclic_schedule_never_overlaps_at_negative_prices() -> None:
    prices = _typical_day()
    prices[(np.arange(288) / 12 >= 10) & (np.arange(288) / 12 < 15)] = -300.0
    d = optimise(prices, SPEC, cyclic=True)  # no wear cost: the LP is tempted to burn energy
    assert not _overlaps(d)
    lp = optimise(prices, SPEC, cyclic=True, method="lp")
    assert lp.revenue(prices) >= d.revenue(prices) - 1e-6
