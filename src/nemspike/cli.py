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


@app.command("backtest-baselines")
def backtest_baselines(config: Path = ConfigOpt) -> None:
    """Walk-forward backtest of the baselines for each configured horizon."""
    import pandas as pd

    from nemspike.evaluation.backtest import run_backtest, summarise
    from nemspike.features.build import build_features
    from nemspike.models.baselines import default_baselines
    from nemspike.pipeline import ARTEFACT_DIR, REPORT_DIR, load_inputs

    cfg = load_config(config)
    prices, weather = load_inputs(cfg)
    (ARTEFACT_DIR / "backtest").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    cols = ["mae", "rmse", "pinball_mean", "coverage_q0.1_q0.9", "share_below_q0.99"]
    cols += ["spike_pr_auc", "spike_precision", "spike_recall"]
    for h in cfg.forecast.horizons_minutes:
        df = build_features(prices, weather, h, cfg)
        preds = run_backtest(df, lambda: default_baselines(cfg.forecast.quantiles), cfg)
        preds.to_parquet(ARTEFACT_DIR / "backtest" / f"baselines_h{h}.parquet", index=False)
        summary = summarise(preds, cfg)
        summary.to_csv(REPORT_DIR / "metrics" / f"baselines_h{h}.csv")
        summarise(preds, cfg, by_fold=True).to_csv(
            REPORT_DIR / "metrics" / f"baselines_h{h}_by_fold.csv"
        )
        typer.echo(
            f"\nHorizon {h} min: {int(summary['n'].iloc[0]):,} test origins, "
            f"{preds['fold'].nunique()} folds"
        )
        with pd.option_context("display.width", 200, "display.precision", 3):
            typer.echo(summary[cols].to_string())


if __name__ == "__main__":
    app()
