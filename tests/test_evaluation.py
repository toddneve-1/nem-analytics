from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from nemspike.config import Config, SpikeConfig, ValidationConfig
from nemspike.evaluation.backtest import run_backtest, summarise
from nemspike.evaluation.labels import spike_labels, spike_threshold
from nemspike.evaluation.metrics import (
    interval_coverage,
    mae,
    pinball_loss,
    rmse,
    share_below,
    spike_metrics,
)
from nemspike.evaluation.splits import split, walk_forward_folds
from nemspike.features.build import TARGET, TARGET_TIME, build_features
from nemspike.models.base import qcol
from nemspike.models.baselines import Climatology, PointBaseline, default_baselines
from tests.conftest import make_prices, make_weather

UTC = "UTC"


# --- splits ----------------------------------------------------------------------------------


def test_folds_for_project_period_are_quarterly_after_one_year() -> None:
    folds = walk_forward_folds(
        pd.Timestamp("2023-01-07 14:05", tz=UTC),  # first origin after the 1-week warm-up
        pd.Timestamp("2026-10-31 14:00", tz=UTC),
        ValidationConfig(initial_train_months=12, test_fold_months=3),
    )
    assert len(folds) == 12  # 2024 x4, 2025 x4, 2026 Q1-Q3, Oct 2026
    # First test fold starts at 00:00 AEST on 1 Jan 2024 = 14:00 UTC on 31 Dec 2023.
    assert folds[0].test_start == pd.Timestamp("2023-12-31 14:00", tz=UTC)
    assert folds[0].label == "2024-01 to 2024-03"
    assert folds[-1].label == "2026-10 to 2026-10"
    for a, b in itertools.pairwise(folds):
        assert a.test_end == b.test_start  # contiguous, no overlap


def test_holdout_is_excluded_from_dev_folds_and_scored_separately() -> None:
    from datetime import date

    first = pd.Timestamp("2023-01-07 14:05", tz=UTC)
    last = pd.Timestamp("2026-10-31 14:00", tz=UTC)
    vcfg = ValidationConfig(
        initial_train_months=12, test_fold_months=3, holdout_start=date(2026, 7, 1)
    )
    dev = walk_forward_folds(first, last, vcfg, "dev")
    holdout_start = pd.Timestamp("2026-06-30 14:00", tz=UTC)  # 00:00 AEST 1 Jul 2026
    assert len(dev) == 10 and dev[-1].label == "2026-04 to 2026-06"
    assert dev[-1].test_end == holdout_start
    (h,) = walk_forward_folds(first, last, vcfg, "holdout")
    assert h.is_holdout and h.test_start == holdout_start
    assert h.label == "holdout 2026-07 to 2026-10"


def _toy_frame(h_minutes: int) -> pd.DataFrame:
    origins = pd.date_range("2023-12-01 14:00", "2024-02-29 14:00", freq="5min", tz=UTC)
    return pd.DataFrame(
        {TARGET: 1.0, TARGET_TIME: origins + pd.Timedelta(minutes=h_minutes)}, index=origins
    )


@pytest.mark.parametrize("h", [30, 240])
def test_split_embargo_keeps_training_targets_before_test(h: int) -> None:
    df = _toy_frame(h)
    folds = walk_forward_folds(
        df.index.min(), df.index.max(), ValidationConfig(initial_train_months=1, test_fold_months=1)
    )
    for fold in folds:
        train, test = split(df, fold)
        assert train[TARGET_TIME].max() <= fold.test_start
        assert train.index.max() < test.index.min()
        # The embargo removes exactly the origins whose target falls inside the test period.
        assert train.index.max() == fold.test_start - pd.Timedelta(minutes=h)


def test_split_drops_unknown_targets() -> None:
    df = _toy_frame(30)
    df.iloc[-6:, df.columns.get_loc(TARGET)] = np.nan
    folds = walk_forward_folds(
        df.index.min(), df.index.max(), ValidationConfig(initial_train_months=1, test_fold_months=1)
    )
    _, test = split(df, folds[-1])
    assert test[TARGET].notna().all()


# --- labels ----------------------------------------------------------------------------------


def test_percentile_threshold_uses_training_targets_only() -> None:
    train = pd.Series(np.arange(1, 101, dtype=float))
    cfg = SpikeConfig(method="percentile", fixed_threshold=300, percentile=99)
    assert spike_threshold(train, cfg) == pytest.approx(np.percentile(train, 99))
    assert (
        spike_threshold(train, SpikeConfig(method="fixed", fixed_threshold=300, percentile=99))
        == 300
    )


def test_spike_label_is_strictly_above_threshold() -> None:
    labels = spike_labels(pd.Series([299.0, 300.0, 300.01, np.nan]), 300.0)
    assert labels.iloc[:3].tolist() == [0.0, 0.0, 1.0] and np.isnan(labels.iloc[3])


# --- metrics ---------------------------------------------------------------------------------


def test_point_metrics() -> None:
    y, p = np.array([0.0, 10.0]), np.array([1.0, 7.0])
    assert mae(y, p) == 2.0
    assert rmse(y, p) == pytest.approx(np.sqrt(5.0))


def test_pinball_loss_asymmetry() -> None:
    # Under-forecast by 10 at q=0.9 costs 9; over-forecast by 10 costs 1.
    assert pinball_loss(np.array([10.0]), np.array([0.0]), 0.9) == pytest.approx(9.0)
    assert pinball_loss(np.array([0.0]), np.array([10.0]), 0.9) == pytest.approx(1.0)
    assert pinball_loss(np.array([3.0]), np.array([1.0]), 0.5) == pytest.approx(1.0)  # = MAE / 2


def test_pinball_is_minimised_by_true_quantile() -> None:
    y = np.random.default_rng(0).standard_t(3, 50_000) * 50
    true_q90 = np.quantile(y, 0.9)
    losses = {
        c: pinball_loss(y, np.full_like(y, c), 0.9)
        for c in (true_q90 - 20, true_q90, true_q90 + 20)
    }
    assert min(losses, key=losses.get) == true_q90


def test_calibration_and_coverage() -> None:
    y = np.arange(100, dtype=float)
    assert share_below(y, np.full(100, 9.0)) == pytest.approx(0.10)
    assert interval_coverage(y, np.full(100, 10.0), np.full(100, 89.0)) == pytest.approx(0.80)


def test_spike_metrics_random_and_perfect() -> None:
    rng = np.random.default_rng(0)
    label = (rng.random(20_000) < 0.01).astype(int)
    random = spike_metrics(label, rng.random(20_000), 0.5)
    assert random["spike_pr_auc"] == pytest.approx(label.mean(), abs=0.01)
    perfect = spike_metrics(label, label.astype(float), 0.5)
    assert perfect["spike_pr_auc"] == 1.0
    assert perfect["spike_precision"] == 1.0 and perfect["spike_recall"] == 1.0


# --- baselines and backtest ------------------------------------------------------------------


@pytest.fixture(scope="module")
def long_frame(cfg: Config) -> pd.DataFrame:
    prices = make_prices(days=75, seed=3)
    return build_features(prices, make_weather(prices), 30, cfg)


def test_persistence_point_is_latest_price(long_frame: pd.DataFrame) -> None:
    model = PointBaseline("persistence", "rrp_lag_0", [0.1, 0.5, 0.9, 0.99])
    fc = model.fit(long_frame.iloc[:5000], 300.0).predict(long_frame.iloc[5000:6000])
    pd.testing.assert_series_equal(
        fc.point, long_frame["rrp_lag_0"].iloc[5000:6000], check_names=False
    )


@pytest.mark.parametrize("model", ["persistence", "climatology"])
def test_baseline_quantiles_are_ordered_and_calibrated_in_sample(
    long_frame: pd.DataFrame, model: str
) -> None:
    qs = [0.1, 0.5, 0.9, 0.99]
    m = PointBaseline("persistence", "rrp_lag_0", qs) if model == "persistence" else Climatology(qs)
    fc = m.fit(long_frame, 300.0).predict(long_frame)
    q = fc.quantiles
    assert (q[qcol(0.1)] <= q[qcol(0.5)]).all() and (q[qcol(0.9)] <= q[qcol(0.99)]).all()
    # In-sample, empirical quantiles are calibrated by construction (up to grouping/ties).
    assert share_below(long_frame[TARGET], q[qcol(0.9)]) == pytest.approx(0.9, abs=0.01)


def test_backtest_end_to_end(long_frame: pd.DataFrame, cfg: Config) -> None:
    small = cfg.model_copy(deep=True)
    small.validation = ValidationConfig(initial_train_months=1, test_fold_months=1)
    preds = run_backtest(long_frame, lambda: default_baselines(small.forecast.quantiles), small)
    assert set(preds["model"]) == {
        "persistence",
        "same_time_yesterday",
        "same_time_last_week",
        "climatology",
    }
    # Every test prediction is out of sample: its fold's training targets all precede it.
    first_test_origin = preds.groupby("fold")["origin"].min()
    assert (first_test_origin.sort_index().diff().dropna() > pd.Timedelta(0)).all()
    summary = summarise(preds, small)
    assert {"mae", "rmse", "pinball_mean", "coverage_q0.1_q0.9", "spike_pr_auc"} <= set(
        summary.columns
    )
    assert (summary["n"] == summary["n"].iloc[0]).all()  # all models scored on the same rows
