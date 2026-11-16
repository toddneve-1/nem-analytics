from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nemspike.config import Config
from nemspike.features.build import TARGET, TARGET_TIME, build_features
from nemspike.models.base import qcol
from nemspike.models.lgbm import LightGBMForecaster, inner_split, subsample
from nemspike.models.transforms import TargetTransform
from tests.conftest import make_prices, make_weather

QS = [0.1, 0.5, 0.9, 0.99]


# --- transforms ------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["none", "asinh"])
def test_transform_round_trip_including_negative_and_extreme(kind: str) -> None:
    t = TargetTransform(kind, 100.0)
    y = np.array([-1000.0, -50.0, 0.0, 50.0, 300.0, 17500.0])
    np.testing.assert_allclose(t.inverse(t.forward(y)), y, rtol=1e-12, atol=1e-9)


def test_asinh_is_monotone_so_quantiles_map_exactly() -> None:
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(50, 40, 10_000), rng.pareto(1.5, 200) * 300])
    t = TargetTransform("asinh", 100.0)
    for q in QS:
        # With method="lower" the quantile is an actual sample, so the identity is exact.
        q_of_transformed = np.quantile(t.forward(y), q, method="lower")
        assert t.inverse(q_of_transformed) == pytest.approx(np.quantile(y, q, method="lower"))


def test_asinh_compresses_tails() -> None:
    t = TargetTransform("asinh", 100.0)
    assert t.forward(np.array([17500.0]))[0] == pytest.approx(5.858, abs=1e-3)
    assert t.forward(np.array([-1000.0]))[0] == pytest.approx(-2.998, abs=1e-3)


# --- fitting helpers -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frame(cfg: Config) -> pd.DataFrame:
    prices = make_prices(days=75, seed=5)
    return build_features(prices, make_weather(prices), 240, cfg).dropna(subset=[TARGET])


def test_subsample_keeps_stride(frame: pd.DataFrame) -> None:
    sub = subsample(frame, 15)
    assert ((sub.index.minute % 15) == 0).all()
    assert len(sub) == pytest.approx(len(frame) / 3, rel=0.01)


def test_inner_split_has_embargo(frame: pd.DataFrame) -> None:
    inner, val = inner_split(frame, validation_days=14)
    assert inner[TARGET_TIME].max() <= val.index.min()
    assert inner.index.max() < val.index.min()


# --- LightGBM forecaster ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cfg(cfg: Config) -> Config:
    c = cfg.model_copy(deep=True)
    c.model.max_rounds = 60
    c.model.early_stopping_rounds = 10
    c.model.validation_days = 14
    c.model.min_data_in_leaf = 20
    c.model.num_threads = 1  # small data; avoid contending with any running backtest
    return c


def _fit_predict(frame: pd.DataFrame, small_cfg: Config, seed: int = 42):
    train, test = frame.iloc[:-2000], frame.iloc[-2000:]
    model = LightGBMForecaster(QS, TargetTransform("asinh", 100.0), small_cfg.model, seed)
    return model, model.fit(train, 300.0).predict(test)


def test_lgbm_outputs_ordered_quantiles_and_probabilities(
    frame: pd.DataFrame, small_cfg: Config
) -> None:
    model, fc = _fit_predict(frame, small_cfg)
    q = fc.quantiles
    assert list(q.columns) == [qcol(x) for x in QS]
    assert (np.diff(q.to_numpy(), axis=1) >= 0).all()  # no crossing after sorting
    assert 0 <= model.crossing_rate <= 1
    assert fc.spike_prob.between(0, 1).all()
    pd.testing.assert_series_equal(fc.point, q[qcol(0.5)])
    assert set(model.best_rounds) == {*(qcol(x) for x in QS), "spike"}


def test_lgbm_is_deterministic_for_a_fixed_seed(frame: pd.DataFrame, small_cfg: Config) -> None:
    _, a = _fit_predict(frame, small_cfg, seed=7)
    _, b = _fit_predict(frame, small_cfg, seed=7)
    pd.testing.assert_frame_equal(a.quantiles, b.quantiles)
    pd.testing.assert_series_equal(a.spike_prob, b.spike_prob)


def test_lgbm_median_beats_constant_forecast(frame: pd.DataFrame, small_cfg: Config) -> None:
    """Sanity: on the synthetic daily cycle the median should beat a constant forecast."""
    _, fc = _fit_predict(frame, small_cfg)
    y = frame[TARGET].iloc[-2000:]
    mae_model = np.mean(np.abs(y - fc.point))
    mae_const = np.mean(np.abs(y - frame[TARGET].iloc[:-2000].median()))
    assert mae_model < mae_const
