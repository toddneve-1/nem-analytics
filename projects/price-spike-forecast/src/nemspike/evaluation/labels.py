"""Spike definition. A percentile threshold is fitted on training targets only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nemspike.config import SpikeConfig


def spike_threshold(train_target: pd.Series, cfg: SpikeConfig) -> float:
    """$/MWh level above which a price counts as a spike.

    With ``method: percentile`` the level is the given percentile of the *training-fold* targets,
    never of the test data, so the label definition cannot leak information from the future.
    """
    if cfg.method == "fixed":
        return float(cfg.fixed_threshold)
    values = train_target.dropna().to_numpy()
    if values.size == 0:
        raise ValueError("cannot fit a percentile threshold on an empty training set")
    return float(np.percentile(values, cfg.percentile))


def spike_labels(target: pd.Series, threshold: float) -> pd.Series:
    """1 if the price is strictly above the threshold, else 0 (NaN targets stay NaN)."""
    return (target > threshold).astype(float).where(target.notna())
