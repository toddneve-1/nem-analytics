"""End-to-end smoke test of the command line on small synthetic data (no network, no MLflow).

Runs the README's pipeline in order: backtest -> battery -> battery-forecast. It checks that the
commands work from a clean directory, that the dispatch settings really come from the config, and
that the fold wiring in ``battery-forecast`` holds together.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from nemspike.cli import app
from nemspike.data import aemo, weather
from tests.conftest import DEFAULT_CONFIG, make_prices, make_weather

DAYS = 90  # 2 Jan - 31 Mar 2024 (synthetic data starts on 2 Jan)


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A working directory with cached synthetic data and a small, fast config."""
    root = tmp_path_factory.mktemp("project")
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["data"].update(start="2024-01-02", end="2024-03-31", raw_dir=str(root / "data" / "raw"))
    raw["forecast"].update(horizons_minutes=[30], dispatch_horizons_minutes=[30, 60])
    raw["validation"] = {"initial_train_months": 1, "test_fold_months": 1}
    raw["model"].update(validation_days=7, max_rounds=10, early_stopping_rounds=5, num_threads=1)
    raw["battery"].update(lookahead_hours=1)
    (root / "cfg.yaml").write_text(yaml.safe_dump(raw))

    prices = make_prices(days=DAYS, seed=3)
    nem = prices["interval_start"].dt.tz_convert("Etc/GMT-10")
    for (y, m), month in prices.groupby([nem.dt.year, nem.dt.month]):
        path = aemo._cache_path(Path(raw["data"]["raw_dir"]), "VIC1", y, m)
        path.parent.mkdir(parents=True, exist_ok=True)
        month.to_parquet(path, index=False)
    w = make_weather(prices)
    wcfg = raw["data"]["weather"]
    for y, year in w.groupby(w["valid_time"].dt.year):
        path = weather._cache_path(
            Path(raw["data"]["raw_dir"]), wcfg["model"], wcfg["latitude"], wcfg["longitude"], y
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        year.to_parquet(path, index=False)
    return root


def _run(project: Path, monkeypatch: pytest.MonkeyPatch, *args: str) -> str:
    monkeypatch.chdir(project)
    result = CliRunner().invoke(app, [*args, "--config", "cfg.yaml"])
    assert result.exit_code == 0, result.output + repr(result.exception)
    return result.output


def test_pipeline_runs_end_to_end(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Before any backtest, battery-forecast must say which command produces what it needs.
    monkeypatch.chdir(project)
    early = CliRunner().invoke(app, ["battery-forecast", "--no-track", "--config", "cfg.yaml"])
    assert isinstance(early.exception, FileNotFoundError)
    assert "nemspike backtest" in str(early.exception)

    _run(project, monkeypatch, "backtest", "--models", "lgbm_asinh", "--horizons", "dispatch",
         "--no-track")  # fmt: skip
    for h in (30, 60):  # exactly the configured dispatch grid
        assert (project / f"artefacts/backtest/dev/h{h}/lgbm_asinh.parquet").exists()
    assert not (project / "artefacts/backtest/dev/h240").exists()

    _run(project, monkeypatch, "battery", "--no-track")
    assert (project / "reports/metrics/dev/battery_summary_continuous.csv").exists()

    _run(project, monkeypatch, "battery-forecast", "--price-methods", "learned", "--execution",
         "plan", "--tag", "forecast", "--no-track")  # fmt: skip
    metrics = project / "reports/metrics/dev"
    summary = pd.read_csv(metrics / "battery_summary_forecast_deg40.csv", index_col=0)
    assert set(summary.index) == {"forecast_learned", "naive_time_of_day", "perfect_foresight"}
    # Fold 0 (February) is skipped, so only March is evaluated, on the same days for everyone.
    assert (summary["days"] == 31).all()
    fc, pf = summary.loc["forecast_learned"], summary.loc["perfect_foresight"]
    assert fc["net_revenue"] <= pf["net_revenue"] + 1e-6
    assert fc["net_vs_naive_time_of_day_ci_low"] <= fc["net_vs_naive_time_of_day"]
    assert fc["net_vs_naive_time_of_day"] <= fc["net_vs_naive_time_of_day_ci_high"]
    # One storage-value cache, named by a fingerprint rather than the wear cost alone.
    assert len(list((project / "artefacts/battery").glob("storage_values_*.parquet"))) == 1

    # The ladder run: threshold execution, recalibrated price and the threshold-only benchmark.
    _run(project, monkeypatch, "battery-forecast", "--price-methods", "learned,recalibrated",
         "--execution", "plan,threshold", "--threshold-only", "--tag", "ladder",
         "--no-track")  # fmt: skip
    ladder = pd.read_csv(metrics / "battery_summary_ladder_deg40.csv", index_col=0)
    assert set(ladder.index) == {
        "forecast_learned", "forecast_recalibrated", "threshold_learned",
        "threshold_recalibrated", "threshold_only", "naive_time_of_day", "perfect_foresight",
    }  # fmt: skip
    assert (ladder["days"] == 31).all()
    assert (ladder["net_revenue"] <= ladder.loc["perfect_foresight", "net_revenue"] + 1e-6).all()
    assert ladder["capture_best10"].notna().all()

    # Step 2: short-horizon point forecasts, block-mean targets, and the fine planning grid.
    _run(project, monkeypatch, "backtest", "--models", "lgbm_asinh", "--horizons", "5,10,15,20,25",
         "--no-track")  # fmt: skip
    _run(project, monkeypatch, "backtest", "--models", "lgbm_asinh", "--horizons", "60",
         "--target", "block_mean", "--no-track")  # fmt: skip
    assert (project / "artefacts/backtest/dev/h60_block/lgbm_asinh.parquet").exists()
    assert (metrics / "h60_block_summary.csv").exists()
    _run(project, monkeypatch, "battery-forecast", "--price-methods", "learned", "--plan-grid",
         "fine", "--execution", "plan,curve", "--tag", "fine", "--no-track")  # fmt: skip
    fine = pd.read_csv(metrics / "battery_summary_fine_deg40.csv", index_col=0)
    assert {"forecast_learned", "curve_learned"} <= set(fine.index)
    assert (fine["net_revenue"] <= fine.loc["perfect_foresight", "net_revenue"] + 1e-6).all()

    # Defaults reproduce the adopted design: learned price, offer-curve execution, tag "curve".
    _run(project, monkeypatch, "battery-forecast", "--no-track")
    adopted = pd.read_csv(metrics / "battery_summary_curve_deg40.csv", index_col=0)
    assert set(adopted.index) == {"curve_learned", "naive_time_of_day", "perfect_foresight"}


def test_battery_forecast_falls_back_to_committed_snapshots(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no artefacts, the compact snapshots under results/predictions are enough."""
    _run(project, monkeypatch, "snapshot-predictions")
    assert (project / "results/predictions/dev/h60_lgbm_asinh.parquet").exists()
    metrics = project / "reports/metrics/dev"
    before = pd.read_csv(metrics / "battery_summary_forecast_deg40.csv", index_col=0)
    (project / "artefacts/backtest").rename(project / "artefacts/backtest_moved")
    try:
        _run(project, monkeypatch, "battery-forecast", "--price-methods", "learned",
             "--execution", "plan", "--tag", "forecast", "--no-track")  # fmt: skip
    finally:
        (project / "artefacts/backtest_moved").rename(project / "artefacts/backtest")
    after = pd.read_csv(metrics / "battery_summary_forecast_deg40.csv", index_col=0)
    # Snapshots are 32-bit, so allow small differences from the full-precision run.
    assert after.loc["forecast_learned", "net_revenue"] == pytest.approx(
        before.loc["forecast_learned", "net_revenue"], rel=0.02
    )
