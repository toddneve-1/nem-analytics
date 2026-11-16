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
