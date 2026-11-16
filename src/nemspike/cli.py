"""Command-line entry point. Sub-commands are added as each pipeline stage is built."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from nemspike.config import load_config

app = typer.Typer(
    help="NEM VIC1 price spike forecasting and battery evaluation.", no_args_is_help=True
)

ConfigOpt = typer.Option(Path("configs/default.yaml"), "--config", "-c", help="YAML config file")


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def version() -> None:
    """Print the package version."""
    from nemspike import __version__

    typer.echo(__version__)


@app.command("fetch-prices")
def fetch_prices(config: Path = ConfigOpt, force: bool = False) -> None:
    """Download and cache AEMO 5-minute price and demand data, then print a quality report."""
    from nemspike.data import aemo

    cfg = load_config(config)
    d = cfg.data
    paths = aemo.fetch_range(d.start, d.end, d.region, d.raw_dir, force=force)
    typer.echo(f"{len(paths)} monthly files cached under {d.raw_dir / 'aemo' / d.region}")
    df = aemo.load_price_demand(d.start, d.end, d.region, d.raw_dir)
    report = aemo.quality_report(df, d.start, d.end)
    typer.echo(report.summary())
    if report.n_missing:
        typer.echo(f"first missing interval ends: {report.missing_interval_ends[:5]}")


@app.command("fetch-weather")
def fetch_weather(config: Path = ConfigOpt, force: bool = False) -> None:
    """Download and cache hourly Open-Meteo weather, then print a quality report."""
    from nemspike.data import weather

    cfg = load_config(config)
    d, w = cfg.data, cfg.data.weather
    paths = weather.fetch_range(
        d.start, d.end, w.latitude, w.longitude, d.raw_dir, model=w.model, force=force
    )
    typer.echo(f"{len(paths)} yearly files cached under {paths[0].parent}")
    df = weather.load_weather(d.start, d.end, w.latitude, w.longitude, d.raw_dir, model=w.model)
    typer.echo(weather.quality_report(df, list(weather.VARIABLES)).summary())


if __name__ == "__main__":
    app()
