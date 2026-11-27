"""Weather years: download ISP traces one reference year at a time, keep only what the model uses.

The 2024 ISP traces cover 13 weather reference years (2011-2023), each mapped by AEMO onto
future dates. Raw files are about 1.7 GB per year, so each year is downloaded from the openISP
mirror, reduced to hourly regional series for the study year, saved under
``inputs/weather/reference_year=<y>/`` (a few MB), and the raw files deleted before the next.

Saved per year:

* ``demand.parquet``: hourly operational demand (MW) by region and POE (``NSW1_POE50`` ...);
* ``vre_cf.parquet``: hourly capacity-weighted wind and solar capacity factors by region;
* ``matching.csv``: how wind and solar units were matched to traces;
* ``manifest.json``: source URLs, file hashes, the unit table used, and the study year.
"""

from __future__ import annotations

import json
import shutil
from importlib.resources import files as package_files
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests

from nemmodel.inputs import ModelYear, sha256
from nemmodel.traces import hourly_demand, regional_vre_capacity_factors

KINDS = ("project/", "zone/", "demand/scenario=Step%20Change/")
PREFIX = "https://data.openisp.au/processed/traces/"


def trace_urls(reference_year: int) -> list[str]:
    """URLs of the processed trace files the model needs for one reference year."""
    manifest = package_files("isp_trace_parser.remote.manifests") / "processed/full_isp_2024.txt"
    urls = [u.strip() for u in manifest.read_text().splitlines() if u.strip()]
    return [u for u in urls if f"reference_year={reference_year}/" in u
            and any(f"/isp_2024/{k}" in u for k in KINDS)]  # fmt: off


def download(urls: list[str], raw_root: Path) -> list[Path]:
    """Download to ``raw_root`` keeping the mirror's folder layout (``isp_2024/...``)."""
    paths = []
    for url in urls:
        dest = raw_root / unquote(url.removeprefix(PREFIX))
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with part.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        part.rename(dest)
        paths.append(dest)
    return paths


def extract(trace_dir: Path, units: pd.DataFrame, year: ModelYear, reference_year: int,
            out_dir: Path, sources: dict) -> None:  # fmt: skip
    """Write the hourly series for one reference year."""
    out_dir.mkdir(parents=True, exist_ok=True)
    demand = hourly_demand(trace_dir, year, "Step Change", reference_year)
    cf, matching = regional_vre_capacity_factors(trace_dir, units, year, reference_year)
    demand.to_parquet(out_dir / "demand.parquet")
    cf.to_parquet(out_dir / "vre_cf.parquet")
    matching.to_csv(out_dir / "matching.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(
        {"reference_year": reference_year, "financial_year": year.label, "scenario": "Step Change",
         "trace_source": "2024 ISP traces (AEMO), via the openISP mirror", **sources},
        indent=2))  # fmt: skip


def fetch_year(reference_year: int, units: pd.DataFrame, units_source: str, year: ModelYear,
               raw_root: Path, out_root: Path, keep_raw: bool = False) -> Path:  # fmt: skip
    """Download, extract and (unless ``keep_raw``) delete one reference year."""
    urls = trace_urls(reference_year)
    if len(urls) != 7:
        raise ValueError(f"expected 7 trace files for {reference_year}, found {len(urls)}")
    paths = download(urls, raw_root)
    out_dir = out_root / f"reference_year={reference_year}"
    sources = {"files": {u: sha256(p) for u, p in zip(urls, paths, strict=True)},
               "units": units_source}  # fmt: skip
    extract(raw_root / "isp_2024", units, year, reference_year, out_dir, sources)
    if not keep_raw:
        for p in paths:
            p.unlink()
        for d in sorted({p.parent for p in paths}, reverse=True):
            shutil.rmtree(d, ignore_errors=True)
    return out_dir


def load_year(out_root: Path, reference_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = out_root / f"reference_year={reference_year}"
    if not d.exists():
        raise FileNotFoundError(f"no weather data for {reference_year}: run nemmodel fetch-weather")
    return pd.read_parquet(d / "demand.parquet"), pd.read_parquet(d / "vre_cf.parquet")
