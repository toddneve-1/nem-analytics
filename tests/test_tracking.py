from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nemspike.config import Config
from nemspike.tracking import mlflow_utils
from nemspike.tracking.mlflow_utils import flatten, log_backtest
from tests.conftest import DEFAULT_CONFIG


def test_flatten_nested_config(cfg: Config) -> None:
    flat = flatten(cfg.model_dump(mode="json"))
    assert flat["seed"] == "42"
    assert flat["battery.power_mw"] == "100.0"
    assert flat["forecast.quantiles"] == "[0.1, 0.5, 0.9, 0.99]"
    assert all(isinstance(v, str) for v in flat.values())


@pytest.fixture
def tracked_cfg(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    c = cfg.model_copy(deep=True)
    c.tracking.mlflow_tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    c.tracking.experiment_name = "test-experiment"
    monkeypatch.setattr(mlflow_utils, "ARTIFACT_ROOT", tmp_path / "artifacts")
    monkeypatch.chdir(tmp_path)
    return c


def test_log_backtest_records_params_metrics_tags_and_folds(tracked_cfg: Config) -> None:
    import mlflow

    pooled = pd.Series({"mae": 12.5, "pinball_mean": 4.0, "spike_precision": float("nan")})
    by_fold = pd.DataFrame(
        {"mae": [10.0, 15.0], "crossing_rate": [0.01, 0.02]},
        index=pd.Index(["2024-01 to 2024-03", "2024-04 to 2024-06"], name="fold_label"),
    )
    run_id = log_backtest(tracked_cfg, DEFAULT_CONFIG, "dev", 30, "persistence", pooled, by_fold)

    run = mlflow.MlflowClient(tracked_cfg.tracking.mlflow_tracking_uri).get_run(run_id)
    assert run.data.params["seed"] == "42"
    assert run.data.metrics["mae"] == 12.5
    assert "spike_precision" not in run.data.metrics  # NaN is skipped, not logged as garbage
    assert run.data.tags["stage"] == "dev" and run.data.tags["model"] == "persistence"
    assert "git_commit" in run.data.tags
    history = mlflow.MlflowClient(tracked_cfg.tracking.mlflow_tracking_uri).get_metric_history(
        run_id, "fold.mae"
    )
    assert [(m.step, m.value) for m in history] == [(0, 10.0), (1, 15.0)]


def test_train_and_register_creates_loadable_versions(tracked_cfg: Config) -> None:
    from nemspike.features.build import TARGET, build_features
    from nemspike.tracking.registry import load_registered, train_and_register
    from tests.conftest import make_prices, make_weather

    prices = make_prices(days=30, seed=2)
    features = build_features(prices, make_weather(prices), 30, tracked_cfg)
    v1 = train_and_register(tracked_cfg, "persistence", 30, cutoff="end", features=features)
    v2 = train_and_register(tracked_cfg, "persistence", 30, cutoff="end", features=features)
    assert v1.name == "nemspike-vic1-persistence-h30"
    assert (v1.version, v2.version) == ("1", "2")  # each retrain is a new version

    loaded = load_registered(v2.name, v2.version)
    sample = features.dropna(subset=[TARGET]).iloc[-50:]
    out = loaded.predict(sample)
    assert {"q0.1", "q0.5", "q0.9", "q0.99", "point", "spike_prob"} <= set(out.columns)
    pd.testing.assert_series_equal(out["point"], sample["rrp_lag_0"], check_names=False)
