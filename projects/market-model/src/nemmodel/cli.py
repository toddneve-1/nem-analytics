"""Command line: fetch traces, build versioned inputs, run the scenario study, draw figures."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from nemmodel.inputs import ModelYear, write_inputs

app = typer.Typer(no_args_is_help=True, add_completion=False)
ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "data" / "raw" / "traces" / "isp_2024"
WORKBOOK_URL = "https://data.openisp.au/archive/workbooks/7.5.xlsx"


@app.command()
def fetch_traces() -> None:
    """Download the 2024 ISP example traces (reference year 2018, about 0.3 GB) from openISP."""
    from isp_trace_parser.remote import fetch_trace_data

    fetch_trace_data("example", "isp_2024", str(ROOT / "data" / "raw" / "traces"), "processed")


@app.command()
def fetch_weather(
    first: int = 2011, last: int = 2023, keep_raw: bool = False, overwrite: bool = False
) -> None:
    """Download weather reference years from openISP and keep hourly series in inputs/weather/.

    About 1.7 GB is downloaded per year and deleted after extraction (unless --keep-raw)."""
    import pandas as pd
    import yaml

    from nemmodel.weather import fetch_year

    cfg = yaml.safe_load((ROOT / "config" / "assumptions.yaml").read_text())
    units_file = ROOT / cfg["inputs"] / "units.csv"
    units = pd.read_csv(units_file)
    out = ROOT / "inputs" / "weather"
    for ref in range(first, last + 1):
        if (out / f"reference_year={ref}").exists() and not overwrite:
            typer.echo(f"{ref}: already present")
            continue
        fetch_year(ref, units, str(units_file.relative_to(ROOT)),
                   ModelYear(cfg["financial_year_end"]), ROOT / "data" / "raw" / "traces_tmp",
                   out, keep_raw)  # fmt: skip
        typer.echo(f"{ref}: done")


@app.command()
def build_inputs(
    workbook: Path,
    label: str = typer.Option(..., help='Folder name under inputs/, e.g. "final-2026-isp"'),
    year: int = 2029,
    scenario: str = "Step Change",
    source_url: str = WORKBOOK_URL,
) -> None:
    """Parse an IASR workbook into versioned input tables under ``inputs/<label>-v<version>``."""
    from isp_workbook_parser import Parser

    version = Parser(str(workbook)).workbook_version
    out = ROOT / "inputs" / f"{label}-v{version}"
    m = write_inputs(workbook, out, ModelYear(year), scenario, source_url)
    typer.echo(f"wrote {m['units']} units to {out}")


@app.command()
def study(
    samples: int = 0,
    workers: int = 3,
    max_minutes: float = typer.Option(0, help="Stop starting new runs after this; rerun to resume"),
    out: Path = ROOT / "results",
) -> None:
    """Run the 2 x 2 study (coal timing x winter gas) over every weather year, with Monte Carlo
    outages. Resumable: finished runs are saved as they complete and skipped on the next call."""
    from nemmodel.study import run_study

    logging.disable(logging.WARNING)
    eue = run_study(ROOT, out, samples or None, workers, max_minutes or None)
    if eue is not None:
        typer.echo(eue.to_string(index=False))


@app.command()
def figures(results: Path = ROOT / "results") -> None:
    """Draw the README figures from saved results."""
    from nemmodel.figures import draw_all

    draw_all(ROOT, results)


if __name__ == "__main__":
    app()
