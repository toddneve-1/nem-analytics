from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nemspike.config import load_config

DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_config_loads() -> None:
    cfg = load_config(DEFAULT)
    assert cfg.data.region == "VIC1"
    assert cfg.forecast.horizons_minutes == [30, 240]
    assert 0.5 in cfg.forecast.quantiles


def _write(tmp_path: Path, mutate) -> Path:  # type: ignore[no-untyped-def]
    raw = yaml.safe_load(DEFAULT.read_text())
    mutate(raw)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_rejects_non_five_minute_horizon(tmp_path: Path) -> None:
    p = _write(tmp_path, lambda r: r["forecast"].update(horizons_minutes=[7]))
    with pytest.raises(ValidationError):
        load_config(p)


def test_requires_median_quantile(tmp_path: Path) -> None:
    p = _write(tmp_path, lambda r: r["forecast"].update(quantiles=[0.1, 0.9]))
    with pytest.raises(ValidationError):
        load_config(p)


def test_rejects_inconsistent_soc(tmp_path: Path) -> None:
    p = _write(tmp_path, lambda r: r["battery"].update(soc_initial_frac=0.95))
    with pytest.raises(ValidationError):
        load_config(p)


def test_rejects_unknown_keys(tmp_path: Path) -> None:
    """A misspelt optional key must fail, not silently fall back to the default."""
    p = _write(tmp_path, lambda r: r["battery"].update(degradation_cost_per_mhw=10.0))
    with pytest.raises(ValidationError, match="degradation_cost_per_mhw"):
        load_config(p)
    p = _write(tmp_path, lambda r: r.update(batery={}))
    with pytest.raises(ValidationError):
        load_config(p)


def test_dispatch_settings_are_derived_from_the_config() -> None:
    cfg = load_config(DEFAULT)
    assert cfg.battery.resolve_intervals == 6 and cfg.battery.lookahead_intervals == 48
    assert cfg.forecast.all_horizons_minutes == [30, 60, 90, 120, 150, 180, 210, 240]
    assert cfg.forecast.default_model == "lgbm_asinh"


def test_rejects_lookahead_beyond_the_longest_dispatch_horizon(tmp_path: Path) -> None:
    p = _write(tmp_path, lambda r: r["battery"].update(lookahead_hours=6))
    with pytest.raises(ValidationError, match="longest dispatch horizon"):
        load_config(p)


def test_rejects_uneven_dispatch_grid_and_off_grid_replanning(tmp_path: Path) -> None:
    p = _write(tmp_path, lambda r: r["forecast"].update(dispatch_horizons_minutes=[30, 60, 120]))
    with pytest.raises(ValidationError, match="even grid"):
        load_config(p)
    p = _write(tmp_path, lambda r: r["battery"].update(rolling_resolve_minutes=7))
    with pytest.raises(ValidationError):
        load_config(p)
