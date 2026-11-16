"""LightGBM quantile regression and spike classification.

One booster per quantile (pinball loss, ``objective="quantile"``) plus one binary classifier for
P(price > spike threshold). All share the same features and fitting procedure:

1. **Subsample** training origins to every ``train_stride_minutes`` (neighbouring 5-minute rows
   are nearly identical, so this saves time with little information loss).
2. **Early stopping** on the last ``validation_days`` of the training fold. An embargo applies
   inside the training fold too: inner-training rows must have targets realised before the
   validation window starts.
3. **Refit** on the whole (subsampled) training fold with the number of trees found in step 2, so
   the most recent data, which matters most in a non-stationary market, is not left out.

Quantile models train on the transformed target and are mapped back exactly (see
:mod:`nemspike.models.transforms`). Separately trained quantile models can "cross" (for example
q0.9 < q0.5 for some rows); we record how often and then sort each row, which can only reduce
pinball loss.

The classifier is trained without class re-weighting, so its outputs remain probabilities, and is
early-stopped on log loss (PR-AUC is used for reporting).
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from nemspike.config import ModelConfig
from nemspike.evaluation.labels import spike_labels
from nemspike.features.build import TARGET, TARGET_TIME, feature_columns
from nemspike.models.base import Forecast, Forecaster, qcol
from nemspike.models.transforms import TargetTransform

log = logging.getLogger(__name__)


def _base_params(cfg: ModelConfig, seed: int) -> dict:
    return {
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "lambda_l2": cfg.lambda_l2,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }


def subsample(df: pd.DataFrame, stride_minutes: int) -> pd.DataFrame:
    """Keep origins whose minute of the day is a multiple of the stride."""
    minutes = df.index.hour * 60 + df.index.minute
    return df[(minutes % stride_minutes) == 0]


def inner_split(train: pd.DataFrame, validation_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training fold into inner-train and early-stopping validation, with an embargo."""
    val_start = train.index.max() - pd.Timedelta(days=validation_days)
    inner = train[train[TARGET_TIME] <= val_start]
    val = train[train.index > val_start]
    return inner, val


class LightGBMForecaster(Forecaster):
    def __init__(
        self,
        quantiles: list[float],
        transform: TargetTransform,
        cfg: ModelConfig,
        seed: int,
        name: str | None = None,
    ) -> None:
        self.quantiles = sorted(set(quantiles) | {0.5})
        self.transform = transform
        self.cfg = cfg
        self.seed = seed
        self.name = name or f"lgbm_{transform.kind}"
        self.features: list[str] = []
        self.quantile_models: dict[float, lgb.Booster] = {}
        self.spike_model: lgb.Booster | None = None
        self.best_rounds: dict[str, int] = {}
        self.crossing_rate = float("nan")

    def _fit_one(
        self,
        params: dict,
        inner: pd.DataFrame,
        val: pd.DataFrame,
        full: pd.DataFrame,
        y_inner: np.ndarray,
        y_val: np.ndarray,
        y_full: np.ndarray,
        key: str,
    ) -> lgb.Booster:
        dtrain = lgb.Dataset(inner[self.features], y_inner, free_raw_data=False)
        dval = lgb.Dataset(val[self.features], y_val, reference=dtrain, free_raw_data=False)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=self.cfg.max_rounds,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False)],
        )
        rounds = max(booster.best_iteration, 1)
        self.best_rounds[key] = rounds
        dfull = lgb.Dataset(full[self.features], y_full)
        return lgb.train(params, dfull, num_boost_round=rounds)

    def fit(self, train: pd.DataFrame, spike_threshold: float) -> LightGBMForecaster:
        self.features = feature_columns(train)
        full = subsample(train, self.cfg.train_stride_minutes)
        inner, val = inner_split(full, self.cfg.validation_days)
        base = _base_params(self.cfg, self.seed)

        z = {
            name: self.transform.forward(part[TARGET])
            for name, part in (("inner", inner), ("val", val), ("full", full))
        }
        for q in self.quantiles:
            params = base | {"objective": "quantile", "alpha": q, "metric": "quantile"}
            self.quantile_models[q] = self._fit_one(
                params, inner, val, full, z["inner"], z["val"], z["full"], qcol(q)
            )

        labels = {
            name: spike_labels(part[TARGET], spike_threshold).to_numpy()
            for name, part in (("inner", inner), ("val", val), ("full", full))
        }
        # Early-stop on log loss, not average precision: with only a handful of spikes in the
        # validation window, average precision is too noisy to stop on, whereas log loss is a
        # smooth, proper scoring rule that also rewards calibrated probabilities.
        params = base | {"objective": "binary", "metric": "binary_logloss"}
        self.spike_model = self._fit_one(
            params, inner, val, full, labels["inner"], labels["val"], labels["full"], "spike"
        )
        log.info("%s best rounds: %s", self.name, self.best_rounds)
        return self

    def predict(self, test: pd.DataFrame) -> Forecast:
        if not self.quantile_models or self.spike_model is None:
            raise RuntimeError("fit() must be called before predict()")
        x_test = test[self.features]
        raw = np.column_stack(
            [
                self.transform.inverse(self.quantile_models[q].predict(x_test))
                for q in self.quantiles
            ]
        )
        self.crossing_rate = float(np.mean(np.any(np.diff(raw, axis=1) < 0, axis=1)))
        ordered = np.sort(raw, axis=1)
        quantiles = pd.DataFrame(
            ordered, index=test.index, columns=[qcol(q) for q in self.quantiles]
        )
        prob = pd.Series(self.spike_model.predict(x_test), index=test.index)
        return Forecast(point=quantiles[qcol(0.5)], quantiles=quantiles, spike_prob=prob)
