# NEM VIC1 price spike forecasting with battery dispatch evaluation

Probabilistic forecasts of the 5-minute regional reference price (RRP) in Victoria, a price
spike classifier, and a linear-programming battery model that measures how much of the
perfect-foresight arbitrage revenue the forecasts can capture.

> **Status:** work in progress. Results will be added only from reproducible, seeded runs.

## Quick start

```bash
uv sync --extra dev
```

```bash
uv run pytest
```

```bash
uv run ruff check .
```

See `docs/design_notes.md` for design decisions and limitations.
