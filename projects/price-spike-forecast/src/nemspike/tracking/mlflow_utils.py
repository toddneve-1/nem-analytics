"""Local MLflow tracking.

* **Backtest runs.** One MLflow run per (stage, horizon, model) with: the full flattened config as
  parameters, the git commit and whether the working tree was dirty as tags, pooled metrics,
  per-fold metrics as a series (step = fold number), and the config file and per-fold table as
  artefacts. Any README number can be traced to a run, a commit and a seed.
* **Versioned models.** ``nemspike train-final`` fits a model on all data up to a cut-off and
  registers it in the MLflow Model Registry (SQLite backend), so every retrain gets a new version
  number with its training window, commit and metrics attached.

Storage is local: ``mlflow.db`` (runs and registry) and ``mlartifacts/`` (files), both git-ignored.
Browse with ``uv run mlflow ui --backend-store-uri sqlite:///mlflow.db``.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from nemspike.config import Config

ARTIFACT_ROOT = Path("mlartifacts")


def _mlflow():  # imported lazily: MLflow is slow to import and not needed by most commands
    import mlflow

    return mlflow


def setup(cfg: Config) -> None:
    """Point MLflow at the local store and select the experiment."""
    mlflow = _mlflow()
    mlflow.set_tracking_uri(cfg.tracking.mlflow_tracking_uri)
    if mlflow.get_experiment_by_name(cfg.tracking.experiment_name) is None:
        mlflow.create_experiment(
            cfg.tracking.experiment_name, artifact_location=ARTIFACT_ROOT.resolve().as_uri()
        )
    mlflow.set_experiment(cfg.tracking.experiment_name)


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested dict to dotted keys with string values (MLflow parameters)."""
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = str(v)
    return out


def git_state() -> dict[str, str]:
    """Current commit and whether there are uncommitted changes."""

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_dirty": str(bool(run("status", "--porcelain"))),
    }


def _finite(metrics: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items() if v is not None and math.isfinite(float(v))}


def log_backtest(
    cfg: Config,
    config_path: Path,
    stage: str,
    horizon: int,
    model: str,
    pooled: pd.Series,
    by_fold: pd.DataFrame,
    extra_tags: dict[str, str] | None = None,
) -> str:
    """Record one model's backtest. Returns the MLflow run id."""
    mlflow = _mlflow()
    setup(cfg)
    with mlflow.start_run(run_name=f"{stage}-h{horizon}-{model}") as run:
        mlflow.set_tags(
            {"stage": stage, "horizon_minutes": str(horizon), "model": model, "kind": "backtest"}
            | git_state()
            | (extra_tags or {})
        )
        mlflow.log_params(flatten(cfg.model_dump(mode="json")))
        mlflow.log_metrics(_finite(pooled.to_dict()))
        for step, (_, row) in enumerate(by_fold.iterrows()):
            mlflow.log_metrics(
                {f"fold.{k}": v for k, v in _finite(row.to_dict()).items()}, step=step
            )
        mlflow.log_artifact(str(config_path))
        tmp = Path("artefacts") / "tmp" / f"by_fold_{stage}_h{horizon}_{model}.csv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        by_fold.to_csv(tmp)
        mlflow.log_artifact(str(tmp))
        return run.info.run_id


def log_battery(
    cfg: Config,
    config_path: Path,
    stage: str,
    degradation_cost: float,
    summary: pd.DataFrame,
    daily_csv: Path,
    mode: str = "continuous",
) -> str:
    """Record one battery-strategy evaluation (one wear cost). Returns the MLflow run id.

    Metrics are prefixed by strategy, for example ``perfect_foresight.net_revenue``.
    """
    mlflow = _mlflow()
    setup(cfg)
    with mlflow.start_run(run_name=f"{stage}-battery-{mode}-deg{degradation_cost:g}") as run:
        mlflow.set_tags(
            {
                "stage": stage,
                "kind": "battery",
                "battery_mode": mode,
                "degradation_cost_per_mwh": f"{degradation_cost:g}",
            }
            | git_state()
        )
        mlflow.log_params(flatten(cfg.model_dump(mode="json")))
        mlflow.log_param("battery.degradation_cost_used", degradation_cost)
        for strategy, row in summary.iterrows():
            mlflow.log_metrics({f"{strategy}.{k}": v for k, v in _finite(row.to_dict()).items()})
        mlflow.log_artifact(str(config_path))
        mlflow.log_artifact(str(daily_csv))
        return run.info.run_id
