# NEM analytics

Projects on Australia's National Electricity Market (NEM), built with public data from AEMO and
open-source tools. Each project is self-contained, with its own README, tests and design notes.

| Project | What it does | Status |
|---|---|---|
| [Price spike forecasting and battery dispatch](projects/price-spike-forecast/) | Probabilistic 5-minute price forecasts for Victoria (LightGBM quantiles and a spike classifier) and a battery that bids on them; walk-forward evaluation with bootstrap intervals | Development results complete; final holdout pending |
| [NEM market and reliability model](projects/market-model/) | Five-region hourly PyPSA model built from AEMO's ISP inputs: Monte Carlo outages, expected unserved energy against the reliability standard, and a southern winter gas constraint; coal timing x gas scenario study for FY2028-29 | v0.1 |

## Layout

```
projects/
  price-spike-forecast/   forecasting and battery evaluation
  market-model/           five-region market and reliability model (PyPSA)
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
