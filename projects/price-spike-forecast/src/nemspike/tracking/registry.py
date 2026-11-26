"""Train a final model and register it as a new version in the MLflow Model Registry.

A registered model bundles the fitted forecaster (all quantile boosters, the spike classifier and
the feature list) as an MLflow ``pyfunc`` model. ``predict`` takes the feature table from
:func:`nemspike.features.build.build_features` and returns one column per quantile plus
``spike_prob``. Each version is tagged with its training window, spike threshold and git commit.

The default cut-off is the holdout start, so the registered model is the one that would be scored
on the holdout. ``--cutoff end`` trains on all available data (the "production" model).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nemspike.config import Config
from nemspike.evaluation.labels import spike_threshold
from nemspike.features.build import TARGET, TARGET_TIME, build_features
from nemspike.models.registry import make_model
from nemspike.pipeline import ARTEFACT_DIR, load_inputs
from nemspike.timeutils import nem_day_start_utc
from nemspike.tracking.mlflow_utils import _mlflow, flatten, git_state, setup


def _requirements() -> list[str]:
    """Pin the packages the pickled forecaster needs, from the installed versions (faster and
    more reproducible than MLflow's automatic inference)."""
    from importlib.metadata import version

    pkgs = ["lightgbm", "numpy", "pandas", "scikit-learn", "scipy", "holidays", "pydantic"]
    return [f"{p}=={version(p)}" for p in pkgs]


def registered_name(cfg: Config, model: str, horizon: int) -> str:
    return f"nemspike-{cfg.data.region.lower()}-{model}-h{horizon}"


def _pyfunc_class():
    mlflow = _mlflow()

    class ForecasterModel(mlflow.pyfunc.PythonModel):
        """MLflow wrapper around a fitted :class:`nemspike.models.base.Forecaster`."""

        def load_context(self, context):  # type: ignore[no-untyped-def]
            with open(context.artifacts["forecaster"], "rb") as fh:
                self.forecaster = pickle.load(fh)

        def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:  # type: ignore[no-untyped-def]
            fc = self.forecaster.predict(model_input)
            out = (
                fc.quantiles.copy()
                if fc.quantiles is not None
                else pd.DataFrame(index=model_input.index)
            )
            out["point"] = fc.point
            if fc.spike_prob is not None:
                out["spike_prob"] = fc.spike_prob
            return out

    return ForecasterModel


@dataclass
class RegisteredVersion:
    name: str
    version: str
    run_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    n_train: int


def train_and_register(
    cfg: Config,
    model: str,
    horizon: int,
    cutoff: str = "holdout",
    features: pd.DataFrame | None = None,
) -> RegisteredVersion:
    """Fit ``model`` for ``horizon`` on data before the cut-off and register a new version.

    ``features`` (the output of ``build_features``) may be passed in; otherwise it is built from
    the cached data.
    """
    mlflow = _mlflow()
    if features is None:
        prices, weather = load_inputs(cfg)
        features = build_features(prices, weather, horizon, cfg)
    df = features
    if cutoff == "holdout":
        if cfg.validation.holdout_start is None:
            raise ValueError("no holdout configured; use cutoff='end'")
        limit = nem_day_start_utc(cfg.validation.holdout_start)
    elif cutoff == "end":
        limit = df[TARGET_TIME].max()
    else:
        raise ValueError("cutoff must be 'holdout' or 'end'")
    train = df[df[TARGET].notna() & (df[TARGET_TIME] <= limit)]
    threshold = spike_threshold(train[TARGET], cfg.spike)
    forecaster = make_model(model, cfg).fit(train, threshold)

    out_dir = ARTEFACT_DIR / "models" / f"{model}_h{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = out_dir / "forecaster.pkl"
    with blob.open("wb") as fh:
        pickle.dump(forecaster, fh)

    setup(cfg)
    name = registered_name(cfg, model, horizon)
    with mlflow.start_run(run_name=f"train-{model}-h{horizon}") as run:
        tags = {
            "kind": "train_final",
            "model": model,
            "horizon_minutes": str(horizon),
            "cutoff": cutoff,
            "train_start": str(train.index.min()),
            "train_end_target": str(train[TARGET_TIME].max()),
            "spike_threshold": f"{threshold:.2f}",
        } | git_state()
        mlflow.set_tags(tags)
        mlflow.log_params(flatten(cfg.model_dump(mode="json")))
        mlflow.log_metric("n_train", len(train))
        for key, rounds in getattr(forecaster, "best_rounds", {}).items():
            mlflow.log_metric(f"best_rounds.{key}", rounds)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=_pyfunc_class()(),
            artifacts={"forecaster": str(blob)},
            registered_model_name=name,
            pip_requirements=_requirements(),
        )
        version = str(info.registered_model_version)
        client = mlflow.MlflowClient()
        for k, v in tags.items():
            client.set_model_version_tag(name, version, k, v)
        return RegisteredVersion(
            name=name,
            version=version,
            run_id=run.info.run_id,
            train_start=train.index.min(),
            train_end=train[TARGET_TIME].max(),
            n_train=len(train),
        )


def load_registered(name: str, version: str | int = "latest"):  # type: ignore[no-untyped-def]
    """Load a registered model version as an MLflow pyfunc model."""
    mlflow = _mlflow()
    return mlflow.pyfunc.load_model(f"models:/{name}/{version}")


def model_dir() -> Path:
    return ARTEFACT_DIR / "models"
