"""Shared fixtures: a tiny synthetic two-region system and the committed configuration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "assumptions.yaml").read_text())


@pytest.fixture(scope="session")
def committed_units() -> pd.DataFrame:
    return pd.read_csv(ROOT / "inputs" / "final-2026-isp-v7.8" / "units.csv")


def unit(uid: str, region: str, group: str, cap: float, srmc: float, **kw) -> dict:
    base = {
        "unit_id": uid,
        "station": uid,
        "region": region,
        "group": group,
        "capacity_mw": cap,
        "storage_mwh": 0.0,
        "closure_year": np.nan,
        "heat_rate": 10.0 if group in {"ccgt", "ocgt", "recip"} else np.nan,
        "srmc": srmc,
        "for_full": 0.0,
        "for_partial": 0.0,
        "partial_derate": 0.0,
        "mttr_full": np.nan,
        "mttr_partial": np.nan,
        "rating_summer_typical": cap,
        "rating_winter_early": cap,
        "msl_mw": 0.0,
        "ramp_up_mw_min": np.nan,
        "ramp_down_mw_min": np.nan,
        "rating_winter_late": cap,
    }
    base.update(kw)
    return base


@pytest.fixture
def tiny() -> dict:
    """NSW1 with cheap coal and a gas unit, VIC1 with only a gas unit; one 200 MW link."""
    idx = pd.date_range("2028-07-01", periods=48, freq="h")
    units = pd.DataFrame(
        [
            unit("COAL", "NSW1", "coal", 500, 30),
            unit("GASN", "NSW1", "ocgt", 300, 150),
            unit("GASV", "VIC1", "ocgt", 300, 150),
        ]
    )
    ics = pd.DataFrame(
        [
            {
                "flow_path": "VIC-NSW",
                "from": "VIC1",
                "to": "NSW1",
                "fwd_summer": 200,
                "fwd_winter": 200,
                "rev_summer": 200,
                "rev_winter": 200,
            }
        ]
    )
    regions = ["QLD1", "NSW1", "VIC1", "SA1", "TAS1"]
    demand = pd.DataFrame(0.0, index=idx, columns=regions)
    demand["NSW1"] = 600.0
    demand["VIC1"] = 400.0
    cf = pd.DataFrame(index=idx)
    return {"idx": idx, "units": units, "ics": ics, "demand": demand, "cf": cf}
