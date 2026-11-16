"""Target transforms for heavy-tailed, possibly negative prices.

``asinh(price / s)`` is close to linear for |price| << s and close to ``sign * log(2|price| / s)``
for |price| >> s. Unlike a logarithm it is defined for zero and negative prices.

Because the transform is strictly increasing, quantiles map through it exactly:
``Q_q(asinh(Y / s)) = asinh(Q_q(Y) / s)``. Quantile models trained on the transformed target can
therefore be back-transformed with ``s * sinh(.)`` without bias. Means do *not* map through a
non-linear transform, which is one reason the point forecast is the median (q0.5), not a mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetTransform:
    kind: str  # "none" or "asinh"
    scale: float = 100.0

    def forward(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.kind == "none":
            return y
        if self.kind == "asinh":
            return np.arcsinh(y / self.scale)
        raise ValueError(f"unknown transform {self.kind!r}")

    def inverse(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self.kind == "none":
            return z
        if self.kind == "asinh":
            return self.scale * np.sinh(z)
        raise ValueError(f"unknown transform {self.kind!r}")
