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
def build_inputs(
    workbook: Path, year: int = 2029, scenario: str = "Step Change", source_url: str = WORKBOOK_URL
) -> None:
    """Parse an IASR workbook into versioned input tables under ``inputs/``."""
    from isp_workbook_parser import Parser

    version = Parser(str(workbook)).workbook_version
    out = ROOT / "inputs" / f"draft-2026-isp-v{version}"
    m = write_inputs(workbook, out, ModelYear(year), scenario, source_url)
    typer.echo(f"wrote {m['units']} units to {out}")


@app.command()
def study(samples: int = 0, workers: int = 4, out: Path = ROOT / "results") -> None:
    """Run the 2 x 2 study (coal timing x winter gas) with Monte Carlo outages."""
    from nemmodel.study import run_study

    logging.disable(logging.WARNING)
    eue = run_study(ROOT, TRACES, out, samples or None, workers)
    typer.echo(eue.to_string(index=False))


@app.command()
def figures(results: Path = ROOT / "results") -> None:
    """Draw the README figures from saved results."""
    from nemmodel.figures import draw_all

    draw_all(ROOT, results)


if __name__ == "__main__":
    app()
