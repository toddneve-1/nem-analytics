"""Forecast metrics.

* **MAE / RMSE** on the point (median) forecast. Prices are heavy-tailed, so RMSE is dominated by
  a handful of spikes; MAE is more representative of typical error. Both are reported.
* **Pinball (quantile) loss** for each quantile q: the proper scoring rule for quantile
  forecasts. It is minimised in expectation by the true q-quantile, so lower is better.
* **Calibration**: the share of outcomes at or below each predicted quantile (ideally q) and the
  coverage of the central interval (ideally the nominal level, for example 80 % for q10-q90).
* **Spike classification**: precision and recall at a decision probability, PR-AUC (average
  precision) and the Brier score. With spikes under 1 % of intervals, accuracy and ROC-AUC look
  flattering; PR-AUC is judged against the base rate (a random classifier scores the base rate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Mean quantile loss: q * (y - pred) if y >= pred, else (1 - q) * (pred - y)."""
    diff = np.asarray(y) - np.asarray(pred)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def share_below(y: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of outcomes at or below the predicted quantile (ideally equal to q)."""
    return float(np.mean(np.asarray(y) <= np.asarray(pred)))


def interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def spike_metrics(
    label: np.ndarray, prob: np.ndarray, decision_probability: float
) -> dict[str, float]:
    """Precision, recall, PR-AUC, Brier score and base rate for spike probabilities."""
    label = np.asarray(label).astype(int)
    prob = np.asarray(prob, dtype=float)
    alert = prob >= decision_probability
    tp = int(np.sum(alert & (label == 1)))
    n_alert, n_pos = int(alert.sum()), int(label.sum())
    return {
        "spike_base_rate": float(label.mean()),
        "spike_precision": tp / n_alert if n_alert else float("nan"),
        "spike_recall": tp / n_pos if n_pos else float("nan"),
        "spike_pr_auc": float(average_precision_score(label, prob)) if n_pos else float("nan"),
        "spike_brier": float(np.mean((prob - label) ** 2)),
        "spike_n_positive": float(n_pos),
        "spike_n_alerts": float(n_alert),
    }


def quantile_columns(df: pd.DataFrame) -> dict[float, str]:
    """Map quantile level -> column for columns named like ``q0.1``."""
    return {
        float(c[1:]): c
        for c in df.columns
        if c.startswith("q") and c[1:].replace(".", "").isdigit()
    }


def forecast_metrics(
    y: pd.Series,
    point: pd.Series,
    quantiles: pd.DataFrame | None,
    spike_label: pd.Series | None,
    spike_prob: pd.Series | None,
    decision_probability: float,
    interval: tuple[float, float] = (0.1, 0.9),
) -> dict[str, float]:
    """All metrics for one model on one set of rows."""
    out: dict[str, float] = {"n": float(len(y)), "mae": mae(y, point), "rmse": rmse(y, point)}
    if quantiles is not None:
        qcols = quantile_columns(quantiles)
        for q, col in sorted(qcols.items()):
            out[f"pinball_q{q:g}"] = pinball_loss(y, quantiles[col], q)
            out[f"share_below_q{q:g}"] = share_below(y, quantiles[col])
        out["pinball_mean"] = float(np.mean([out[f"pinball_q{q:g}"] for q in qcols]))
        lo, hi = interval
        if lo in qcols and hi in qcols:
            out[f"coverage_q{lo:g}_q{hi:g}"] = interval_coverage(
                y, quantiles[qcols[lo]], quantiles[qcols[hi]]
            )
            out[f"width_q{lo:g}_q{hi:g}"] = float(
                np.mean(quantiles[qcols[hi]] - quantiles[qcols[lo]])
            )
    if spike_label is not None and spike_prob is not None:
        out.update(spike_metrics(spike_label, spike_prob, decision_probability))
    return out
