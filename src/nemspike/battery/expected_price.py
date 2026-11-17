"""Expected price from quantile forecasts, for planning battery dispatch.

The expected price is the probability-weighted average of the forecast price distribution (the
average of the quantile function over all probability levels). With quantiles at 0.1, 0.5, 0.9
and 0.99 the shape inside each band and the tails are unknown; see design notes section 9.2 for the
comparison of methods. Implemented here:

* ``learned`` (default, method c): from *earlier* out-of-sample forecasts and outcomes, learn the
  empirical probability of each band (which also corrects mis-calibrated intervals), the average
  position of outcomes inside each interior band, and the average excess beyond q0.1 and q0.99.
* ``linear`` (method a): each band's mean is its midpoint; tails taken at q0.1 and q0.99.
* ``median``: q0.5, for comparison (biased low for right-skewed prices).
* ``recalibrated``: the learned expected price mapped through :class:`SpikeRecalibration`, which
  uses the spike probability to correct the under-prediction of high prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

QUANTILE_COLUMNS = ["q0.1", "q0.5", "q0.9", "q0.99"]
PriceMethod = Literal["learned", "linear", "median", "recalibrated"]


def linear_expected(q: np.ndarray) -> np.ndarray:
    """Method a: band midpoints, tails at the outer quantiles. ``q`` has columns q0.1..q0.99."""
    q1, q5, q9, q99 = q.T
    return 0.1 * q1 + 0.4 * (q1 + q5) / 2 + 0.4 * (q5 + q9) / 2 + 0.09 * (q9 + q99) / 2 + 0.01 * q99


@dataclass
class LearnedBands:
    """Parameters of method c, fitted on earlier out-of-sample forecasts."""

    probs: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.4, 0.4, 0.09, 0.01]))
    positions: np.ndarray = field(default_factory=lambda: np.full(3, 0.5))
    low_excess: float = 0.0
    high_excess: float = 0.0

    @classmethod
    def fit(cls, quantiles: np.ndarray, actual: np.ndarray) -> LearnedBands:
        q, y = np.asarray(quantiles, float), np.asarray(actual, float)
        if len(y) == 0:
            raise ValueError("need earlier forecasts and outcomes to fit the learned method")
        band = np.select(
            [y < q[:, 0], y < q[:, 1], y < q[:, 2], y <= q[:, 3]], [0, 1, 2, 3], default=4
        )
        probs = np.bincount(band, minlength=5) / len(y)
        positions = np.full(3, 0.5)
        for j in range(3):
            lo, hi = q[:, j], q[:, j + 1]
            m = (band == j + 1) & (hi - lo > 1e-6)
            if m.any():
                positions[j] = float(np.mean((y[m] - lo[m]) / (hi[m] - lo[m])))
        low = float(np.mean(y[band == 0] - q[band == 0, 0])) if (band == 0).any() else 0.0
        high = float(np.mean(y[band == 4] - q[band == 4, 3])) if (band == 4).any() else 0.0
        return cls(probs=probs, positions=positions, low_excess=low, high_excess=high)

    def expected(self, quantiles: np.ndarray) -> np.ndarray:
        q = np.asarray(quantiles, float)
        means = [q[:, 0] + self.low_excess]
        means += [q[:, j] + self.positions[j] * (q[:, j + 1] - q[:, j]) for j in range(3)]
        means += [q[:, 3] + self.high_excess]
        return sum(self.probs[i] * means[i] for i in range(5))


def expected_price(
    quantiles: pd.DataFrame | np.ndarray,
    method: PriceMethod,
    learned: LearnedBands | None = None,
    recalibration: SpikeRecalibration | None = None,
    spike_prob: np.ndarray | None = None,
) -> np.ndarray:
    """Planning price per row from its quantile forecasts (and spike probability if needed)."""
    q = quantiles[QUANTILE_COLUMNS].to_numpy() if isinstance(quantiles, pd.DataFrame) else quantiles
    if method == "median":
        return np.asarray(q[:, 1], float)
    if method == "linear":
        return linear_expected(q)
    if method == "learned":
        if learned is None:
            raise ValueError("the learned method needs fitted LearnedBands")
        return learned.expected(q)
    if method == "recalibrated":
        if learned is None or recalibration is None or spike_prob is None:
            raise ValueError("recalibrated needs LearnedBands, SpikeRecalibration and spike_prob")
        return recalibration.predict(learned.expected(q), spike_prob)
    raise ValueError(f"unknown price method {method!r}")


# --- recalibration using the spike probability ------------------------------------------------

SPIKE_BUCKET_EDGES = (0.0, 0.05, 0.15, 0.35, 1.0)


@dataclass
class SpikeRecalibration:
    """Map (expected price, spike probability) to the realised average price, learned from
    earlier out-of-sample forecasts.

    Expected prices built from the quantiles under-predict high prices (calibration slopes of
    1.3-1.5; design notes 9.2), and the spike classifier's probability carries information the
    quantiles miss. Within each spike-probability bucket an isotonic (monotone increasing)
    regression of the realised price on the expected price gives the recalibrated planning price.
    Buckets with fewer than ``min_rows`` earlier forecasts are merged into the one below.
    """

    edges: tuple[float, ...] = SPIKE_BUCKET_EDGES
    min_rows: int = 500
    models: dict[int, object] = field(default_factory=dict)
    bucket_map: dict[int, int] = field(default_factory=dict)

    def _bucket(self, spike_prob: np.ndarray) -> np.ndarray:
        inner = np.asarray(self.edges[1:-1])
        return np.searchsorted(inner, np.asarray(spike_prob, float), side="right")

    def fit(
        self, expected: np.ndarray, spike_prob: np.ndarray, actual: np.ndarray
    ) -> SpikeRecalibration:
        from sklearn.isotonic import IsotonicRegression

        e, y = np.asarray(expected, float), np.asarray(actual, float)
        b = self._bucket(spike_prob)
        n_buckets = len(self.edges) - 1
        # Merge sparse buckets downwards (bucket 0, the most populous, always stands alone).
        target = list(range(n_buckets))
        for k in range(1, n_buckets):
            if np.sum(b == k) < self.min_rows:
                target[k] = target[k - 1]
        self.bucket_map = dict(enumerate(target))
        merged = np.vectorize(self.bucket_map.get)(b)
        for k in sorted(set(target)):
            m = merged == k
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            self.models[k] = iso.fit(e[m], y[m])
        return self

    def predict(self, expected: np.ndarray, spike_prob: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("fit() must be called before predict()")
        e = np.asarray(expected, float)
        merged = np.vectorize(self.bucket_map.get)(self._bucket(spike_prob))
        out = np.empty_like(e)
        for k, model in self.models.items():
            m = merged == k
            if m.any():
                out[m] = model.predict(e[m])  # type: ignore[attr-defined]
        return out
