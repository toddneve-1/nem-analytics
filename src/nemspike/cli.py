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


SUMMARY_COLUMNS = [
    "mae",
    "rmse",
    "pinball_mean",
    "pinball_q0.99",
    "coverage_q0.1_q0.9",
    "share_below_q0.99",
    "spike_pr_auc",
    "spike_precision",
    "spike_recall",
    "spike_brier",
]


@app.command()
def backtest(
    config: Path = ConfigOpt,
    models: str = typer.Option("baselines", help="Comma-separated model names or 'baselines'"),
    stage: str = typer.Option("dev", help="'dev' (development folds) or 'holdout' (final test)"),
    horizons: str = typer.Option("", help="Comma-separated horizons in minutes (default: config)"),
) -> None:
    """Walk-forward backtest. Saves predictions per model, then a summary of all saved models."""
    import pandas as pd

    from nemspike.evaluation.backtest import run_backtest, summarise
    from nemspike.features.build import build_features
    from nemspike.models.registry import expand, make_model
    from nemspike.pipeline import ARTEFACT_DIR, REPORT_DIR, load_inputs

    if stage not in ("dev", "holdout"):
        raise typer.BadParameter("stage must be 'dev' or 'holdout'")
    if stage == "holdout":
        typer.echo("Scoring the FINAL HOLDOUT. Do not tune anything on these results.")
    cfg = load_config(config)
    names = expand([m.strip() for m in models.split(",") if m.strip()])
    hs = [int(h) for h in horizons.split(",")] if horizons else cfg.forecast.horizons_minutes
    prices, weather = load_inputs(cfg)

    for h in hs:
        pred_dir = ARTEFACT_DIR / "backtest" / stage / f"h{h}"
        metric_dir = REPORT_DIR / "metrics" / stage
        pred_dir.mkdir(parents=True, exist_ok=True)
        metric_dir.mkdir(parents=True, exist_ok=True)
        df = build_features(prices, weather, h, cfg)
        for name in names:
            typer.echo(f"h={h} min: running {name}")
            preds = run_backtest(df, lambda name=name: [make_model(name, cfg)], cfg, stage)
            preds.to_parquet(pred_dir / f"{name}.parquet", index=False)

        # Summarise every model saved for this stage and horizon, so slow and fast models can be
        # run separately but compared on the same rows.
        all_preds = pd.concat([pd.read_parquet(p) for p in sorted(pred_dir.glob("*.parquet"))])
        summary = summarise(all_preds, cfg)
        summary.to_csv(metric_dir / f"h{h}_summary.csv")
        summarise(all_preds, cfg, by_fold=True).to_csv(metric_dir / f"h{h}_by_fold.csv")
        typer.echo(
            f"\n[{stage}] horizon {h} min: {int(summary['n'].iloc[0]):,} test origins, "
            f"{all_preds['fold'].nunique()} fold(s)"
        )
        with pd.option_context("display.width", 250, "display.precision", 3):
            typer.echo(summary[[c for c in SUMMARY_COLUMNS if c in summary]].to_string())


if __name__ == "__main__":
    app()
