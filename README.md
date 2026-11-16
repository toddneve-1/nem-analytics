# NEM VIC1 price spike forecasting with battery dispatch evaluation

Probabilistic forecasts of the 5-minute regional reference price (RRP) in Victoria, a price
spike classifier, and a linear-programming battery model that measures how much of the
perfect-foresight arbitrage revenue the forecasts can capture.

> **Status:** work in progress. Results will be added only from reproducible, seeded runs.

## Data

| Data | Source | Resolution | Period |
|---|---|---|---|
| Regional reference price and total demand, VIC1 | AEMO aggregated price and demand data | 5 min | 1 Jan 2023 – 31 Oct 2026 |
| Historical weather, Melbourne | Open-Meteo Historical Weather API (ERA5 reanalysis) | Hourly | Same |

Data is not committed to this repository. It is downloaded from the original sources and cached
locally as parquet (see *Quick start*).

## Quick start

```bash
uv sync --extra dev
```

```bash
uv run nemspike fetch-prices
```

```bash
uv run pytest
```

```bash
uv run ruff check .
```

See `docs/design_notes.md` for design decisions and limitations.

## Data attribution and licences

- **AEMO.** Price and demand data © Australian Energy Market Operator (AEMO), used in accordance
  with AEMO's [Copyright Permissions Notice](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions).
  Source: [Aggregated price and demand data](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data).
  This project is not affiliated with or endorsed by AEMO.
- **Open-Meteo.** [Weather data by Open-Meteo.com](https://open-meteo.com/), licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Open-Meteo's free API is for
  non-commercial use only ([terms](https://open-meteo.com/en/terms)). Historical data is derived
  from the Copernicus Climate Change Service (C3S) ERA5 reanalysis. Data has been resampled and
  transformed into features for this project.

## Licence

Code: MIT.
