"""Build forecasters by name, so CLI commands and experiments can select models from the config."""

from __future__ import annotations

from nemspike.config import Config
from nemspike.models.base import Forecaster
from nemspike.models.baselines import default_baselines
from nemspike.models.lgbm import LightGBMForecaster
from nemspike.models.transforms import TargetTransform

MODEL_GROUPS = {
    "baselines": ["persistence", "same_time_yesterday", "same_time_last_week", "climatology"],
}
KNOWN = {*MODEL_GROUPS["baselines"], "lgbm_asinh", "lgbm_none"}


def expand(names: list[str]) -> list[str]:
    """Expand group names (for example ``baselines``) and check every name is known."""
    out: list[str] = []
    for n in names:
        out.extend(MODEL_GROUPS.get(n, [n]))
    unknown = set(out) - KNOWN
    if unknown:
        raise ValueError(f"unknown models {sorted(unknown)}; known: {sorted(KNOWN)} or 'baselines'")
    return out


def make_model(name: str, cfg: Config) -> Forecaster:
    """A fresh, unfitted forecaster."""
    qs = cfg.forecast.quantiles
    if name in MODEL_GROUPS["baselines"]:
        return next(m for m in default_baselines(qs) if m.name == name)
    if name.startswith("lgbm_"):
        kind = name.removeprefix("lgbm_")
        return LightGBMForecaster(
            qs, TargetTransform(kind, cfg.forecast.asinh_scale), cfg.model, cfg.seed, name=name
        )
    raise ValueError(f"unknown model {name!r}")
