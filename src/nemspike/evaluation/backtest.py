"""Walk-forward backtest: fit each forecaster per fold, collect out-of-sample predictions, score.

Predictions are stored in long format (one row per origin per model) so any metric can be
recomputed later without re-running models.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from nemspike.config import Config
from nemspike.evaluation.labels import spike_labels, spike_threshold
from nemspike.evaluation.metrics import forecast_metrics, quantile_columns
from nemspike.evaluation.splits import Fold, Stage, split, walk_forward_folds
from nemspike.features.build import TARGET, TARGET_TIME
from nemspike.models.base import Forecaster

log = logging.getLogger(__name__)

ForecasterFactory = Callable[[], list[Forecaster]]


def run_backtest(
    df: pd.DataFrame,
    make_models: ForecasterFactory,
    cfg: Config,
    stage: Stage = "dev",
    diagnostics: list[dict] | None = None,
) -> pd.DataFrame:
    """Out-of-sample predictions for every fold and model.

    ``make_models`` returns fresh, unfitted forecasters, so no state leaks between folds. If
    ``diagnostics`` is given, per-fold model diagnostics (for example the quantile crossing rate
    and early-stopping rounds of LightGBM) are appended to it.
    """
    folds = walk_forward_folds(df.index.min(), df.index.max(), cfg.validation, stage)
    frames = []
    for fold in folds:
        train, test = split(df, fold)
        threshold = spike_threshold(train[TARGET], cfg.spike)
        label = spike_labels(test[TARGET], threshold)
        log.info("fold %s: %s train rows, %s test rows", fold.label, len(train), len(test))
        for model in make_models():
            fc = model.fit(train, threshold).predict(test)
            if diagnostics is not None:
                diag = {"fold": fold.number, "fold_label": fold.label, "model": model.name}
                if hasattr(model, "crossing_rate"):
                    diag["crossing_rate"] = model.crossing_rate
                for key, rounds in getattr(model, "best_rounds", {}).items():
                    diag[f"best_rounds.{key}"] = rounds
                diagnostics.append(diag)
            out = pd.DataFrame(
                {
                    "fold": fold.number,
                    "fold_label": fold.label,
                    "model": model.name,
                    TARGET_TIME: test[TARGET_TIME],
                    "y": test[TARGET],
                    "point": fc.point,
                    "spike_threshold": threshold,
                    "spike_label": label,
                    "spike_prob": fc.spike_prob if fc.spike_prob is not None else np.nan,
                },
                index=test.index,
            )
            if fc.quantiles is not None:
                out = out.join(fc.quantiles)
            frames.append(out.reset_index())
    return pd.concat(frames, ignore_index=True)


def _score(group: pd.DataFrame, decision_probability: float) -> pd.Series:
    qcols = list(quantile_columns(group).values())
    has_q = bool(qcols) and group[qcols].notna().all().all()
    has_spike = group["spike_prob"].notna().all()
    return pd.Series(
        forecast_metrics(
            group["y"],
            group["point"],
            group[qcols] if has_q else None,
            group["spike_label"] if has_spike else None,
            group["spike_prob"] if has_spike else None,
            decision_probability,
        )
    )


def summarise(predictions: pd.DataFrame, cfg: Config, by_fold: bool = False) -> pd.DataFrame:
    """Metrics per model, pooled over all test folds (or per fold with ``by_fold``)."""
    keys = ["model", "fold_label"] if by_fold else ["model"]
    rows = {
        key: _score(g, cfg.spike.decision_probability)
        for key, g in predictions.groupby(keys, sort=False)
    }
    out = pd.DataFrame(rows).T
    out.index.names = keys
    return out


def folds_for(df: pd.DataFrame, cfg: Config, stage: Stage = "dev") -> list[Fold]:
    return walk_forward_folds(df.index.min(), df.index.max(), cfg.validation, stage)


SNAPSHOT_COLUMNS = ["origin", "fold", "model", "y", "point", "spike_prob"]


def compact_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """A small copy of saved predictions for committing to the repository.

    Keeps the forecast origin, fold, model, actual price, point forecast, quantiles and spike
    probability, as 32-bit floats. Fold labels, target times and spike labels can be recomputed
    from these and the config.
    """
    qcols = list(quantile_columns(predictions).values())
    out = predictions[SNAPSHOT_COLUMNS + qcols].copy()
    for c in ["y", "point", "spike_prob", *qcols]:
        out[c] = out[c].astype("float32")
    out["fold"] = out["fold"].astype("int8")
    out["model"] = out["model"].astype("category")
    return out
