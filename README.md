# NEM analytics

Projects on Australia's National Electricity Market (NEM), built with public data from AEMO and
open-source tools. Each project is self-contained, with its own README, tests and design notes.

| Project | What it does | Status |
|---|---|---|
| [Price spike forecasting and battery dispatch](projects/price-spike-forecast/) | Probabilistic 5-minute price forecasts for Victoria (LightGBM quantiles and a spike classifier) and a battery that bids on them; walk-forward evaluation with bootstrap intervals | Development results complete; final holdout pending |

## Layout

```
projects/
  price-spike-forecast/   forecasting and battery evaluation
```

The repository is a [uv](https://docs.astral.sh/uv/) workspace: one lock file, one environment
per project. From a project's folder:

```bash
uv sync --package <project-package> --extra dev
```

```bash
uv run pytest
```

Each project has its own CI workflow, run when that project's files change.

## Data

Data is never committed. Each project downloads what it needs from the original public sources
and records the source, licence and attribution in its own README.
