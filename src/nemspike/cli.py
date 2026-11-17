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
    track: bool = typer.Option(True, help="Log each model's run to MLflow"),
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
            diagnostics: list[dict] = []
            preds = run_backtest(
                df, lambda name=name: [make_model(name, cfg)], cfg, stage, diagnostics
            )
            preds.to_parquet(pred_dir / f"{name}.parquet", index=False)
            diag = pd.DataFrame(diagnostics).set_index("fold_label").drop(columns=["model"])
            if diag.shape[1] > 1:
                diag.to_csv(metric_dir / f"h{h}_{name}_diagnostics.csv")
            if track:
                from nemspike.tracking.mlflow_utils import log_backtest

                by_fold = summarise(preds, cfg, by_fold=True).droplevel("model")
                by_fold = by_fold.join(diag.drop(columns=["fold"]), how="left")
                run_id = log_backtest(
                    cfg, config, stage, h, name, summarise(preds, cfg).iloc[0], by_fold
                )
                typer.echo(f"  logged to MLflow run {run_id}")

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


@app.command("snapshot-predictions")
def snapshot_predictions(
    models: str = typer.Option("lgbm_asinh", help="Comma-separated model names"),
    stage: str = typer.Option("dev"),
) -> None:
    """Write compact copies of saved backtest predictions to results/predictions (committed)."""
    import pandas as pd

    from nemspike.evaluation.backtest import compact_predictions
    from nemspike.pipeline import ARTEFACT_DIR

    out_dir = Path("results") / "predictions" / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    for h_dir in sorted((ARTEFACT_DIR / "backtest" / stage).glob("h*")):
        for name in [m.strip() for m in models.split(",") if m.strip()]:
            src = h_dir / f"{name}.parquet"
            if not src.exists():
                continue
            dst = out_dir / f"{h_dir.name}_{name}.parquet"
            compact_predictions(pd.read_parquet(src)).to_parquet(
                dst, index=False, compression="zstd", compression_level=19
            )
            typer.echo(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


@app.command("train-final")
def train_final(
    config: Path = ConfigOpt,
    models: str = typer.Option("lgbm_asinh", help="Comma-separated model names"),
    horizons: str = typer.Option("", help="Comma-separated horizons in minutes (default: config)"),
    cutoff: str = typer.Option("holdout", help="'holdout' (train before holdout) or 'end'"),
) -> None:
    """Fit final models and register each as a new version in the MLflow Model Registry."""
    from nemspike.models.registry import expand
    from nemspike.tracking.registry import train_and_register

    cfg = load_config(config)
    hs = [int(h) for h in horizons.split(",")] if horizons else cfg.forecast.horizons_minutes
    for name in expand([m.strip() for m in models.split(",") if m.strip()]):
        for h in hs:
            v = train_and_register(cfg, name, h, cutoff)
            typer.echo(
                f"registered {v.name} version {v.version} "
                f"(trained on {v.n_train:,} rows, targets to {v.train_end}; run {v.run_id})"
            )


@app.command()
def battery(
    config: Path = ConfigOpt,
    stage: str = typer.Option("dev", help="'dev' (development folds) or 'holdout' (final test)"),
    degradation_costs: str = typer.Option(
        "", help="Comma-separated $/MWh wear costs for a sensitivity run (default: config value)"
    ),
    mode: str = typer.Option("", help="'continuous' or 'daily' (default: config value)"),
    track: bool = typer.Option(True, help="Log each wear-cost run to MLflow"),
) -> None:
    """Battery revenue for perfect foresight and the naive time-of-day strategy."""
    from dataclasses import replace

    import pandas as pd

    from nemspike.battery.lp import BatterySpec
    from nemspike.battery.strategies import (
        daily_results,
        naive_time_of_day,
        perfect_foresight,
        summarise_strategies,
    )
    from nemspike.data.aemo import load_price_demand
    from nemspike.evaluation.splits import walk_forward_folds
    from nemspike.pipeline import ARTEFACT_DIR, REPORT_DIR

    if stage not in ("dev", "holdout"):
        raise typer.BadParameter("stage must be 'dev' or 'holdout'")
    cfg = load_config(config)
    d = cfg.data
    base_spec = BatterySpec.from_config(cfg.battery)
    mode = mode or cfg.battery.mode
    if mode not in ("continuous", "daily"):
        raise typer.BadParameter("mode must be 'continuous' or 'daily'")
    costs = (
        [float(c) for c in degradation_costs.split(",")]
        if degradation_costs
        else [base_spec.degradation_cost_per_mwh]
    )
    prices = load_price_demand(d.start, d.end, d.region, d.raw_dir)
    ends = prices["interval_end"]
    folds = walk_forward_folds(ends.min(), ends.max(), cfg.validation, stage)  # type: ignore[arg-type]
    out_dir = ARTEFACT_DIR / "battery" / stage
    metric_dir = REPORT_DIR / "metrics" / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for cost in costs:
        spec = replace(base_spec, degradation_cost_per_mwh=cost)
        frames = []
        for fold in folds:
            # Intervals whose start lies in the fold's test period.
            test = prices[(ends > fold.test_start) & (ends <= fold.test_end)]
            train = prices[ends <= fold.test_start]
            frames.append(perfect_foresight(test, spec, mode).assign(fold_label=fold.label))
            frames.append(naive_time_of_day(train, test, spec).assign(fold_label=fold.label))
        dispatch = pd.concat(frames, ignore_index=True)
        tag = f"{mode}_deg{cost:g}"
        dispatch.to_parquet(out_dir / f"dispatch_{tag}.parquet", index=False)
        daily = daily_results(dispatch, spec)
        daily_csv = metric_dir / f"battery_daily_{tag}.csv"
        daily.to_csv(daily_csv, index=False)
        one = summarise_strategies(daily, spec)
        summaries.append(one.assign(degradation_cost_per_mwh=cost))
        typer.echo(
            f"[{mode}] degradation cost ${cost:g}/MWh: done ({daily['nem_day'].nunique()} days)"
        )
        if track:
            from nemspike.tracking.mlflow_utils import log_battery

            run_id = log_battery(cfg, config, stage, cost, one, daily_csv, mode)
            typer.echo(f"  logged to MLflow run {run_id}")

    summary = pd.concat(summaries).set_index("degradation_cost_per_mwh", append=True)
    summary = summary.reorder_levels(["degradation_cost_per_mwh", "strategy"]).sort_index()
    summary.to_csv(metric_dir / f"battery_summary_{mode}.csv")
    with pd.option_context("display.width", 250, "display.precision", 3):
        typer.echo(summary.to_string())


@app.command()
def figures(config: Path = ConfigOpt) -> None:
    """Write the README figures (data overview, forecast skill by horizon) to reports/figures."""
    from nemspike.data.aemo import load_price_demand
    from nemspike.pipeline import REPORT_DIR
    from nemspike.reporting.figures import data_figures

    cfg = load_config(config)
    d = cfg.data
    prices = load_price_demand(d.start, d.end, d.region, d.raw_dir)
    for path in data_figures(prices, REPORT_DIR / "figures"):
        typer.echo(f"wrote {path}")

    from nemspike.reporting.figures import horizon_figure, horizon_summary

    summary = horizon_summary(REPORT_DIR / "metrics" / "dev")
    if len(summary) >= 2:
        path = horizon_figure(summary, REPORT_DIR / "figures" / "skill_by_horizon.png")
        typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
