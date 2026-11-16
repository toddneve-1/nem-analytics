# Design notes

A running record of design decisions, the reasoning behind them, and known limitations.
Each entry is written to be explainable in an interview: *what* was decided, *why*, and
*what the alternative would have been*.

---

## 1. Project scaffold

**Decision.** `src/` layout package (`nemspike`), configured by `pyproject.toml`, managed with
`uv`. One YAML config (`configs/default.yaml`) validated by pydantic drives every stage.

**Why.**
- The `src/` layout forces tests to run against the installed package, not stray files in the
  working directory, which catches packaging mistakes early.
- A single validated config means every result can be tied to exact settings plus a seed. A bad
  value (for example a 7-minute horizon in a 5-minute market, or an initial state of charge
  outside its limits) fails at load time rather than silently changing the experiment.
- Raw data, MLflow runs and model artefacts are git-ignored: they are reproducible from code,
  and committing data invites stale or hand-edited inputs.

**CI.** GitHub Actions runs `ruff check`, `ruff format --check` and `pytest` on Python 3.11 and
3.12. Tests use small synthetic fixtures; any test that touches the network is marked
`network` and skipped in CI, so CI is deterministic and does not depend on AEMO's servers.

**Default experiment settings (agreed up front).**

| Setting | Value | Note |
|---|---|---|
| Region | VIC1 | |
| Period | 2023-01-01 to 2026-10-31 | Full months, post 5-minute settlement |
| Horizons | 30 min, 4 h | Plus a 30-minute grid to 4 h for dispatch |
| Quantiles | 0.1, 0.5, 0.9, 0.99 | |
| Spike | > $300/MWh (fixed) | Percentile option, fitted on training fold only |
| Battery | 100 MW / 200 MWh, 90 % round trip, SoC 10–90 % | No degradation cost in v1 |
| Seed | 42 | |

---

## 2. Time handling and the AEMO price and demand loader

**Source.** AEMO's monthly *Aggregated price and demand* CSVs
(`PRICE_AND_DEMAND_{YYYYMM}_VIC1.csv`), not NEMOSIS or raw NEMWEB `DISPATCHPRICE` tables.

**Why this source.** The aggregated files give exactly what is needed (5-minute RRP and regional
total demand) in one small file per month (~0.4 MB). NEMWEB's raw MMS tables are much larger and
need more parsing for the same two columns. NEMOSIS remains an option if later features need
other tables (for example interconnector flows or pre-dispatch).

**Timestamps.**
- `SETTLEMENTDATE` labels the **end** of each 5-minute interval in NEM time. `2024/01/01 00:05`
  is the interval 00:00–00:05. We keep both `interval_start` and `interval_end`.
- NEM time is fixed AEST (UTC+10) with **no daylight saving**. We localise with the fixed-offset
  zone `Etc/GMT-10` (the sign is inverted in POSIX `Etc` names), never `Australia/Melbourne`.
  Using Melbourne civil time would shift every summer interval by an hour, which a test documents.
- Everything is stored in **UTC**. Open-Meteo returns UTC, so the weather join cannot be off by
  an hour, and there are no ambiguous or missing hours at daylight-saving changes.
- A "NEM day" is the 288 intervals ending 00:05 to 24:00 AEST. Month files follow the same rule.

**Data quality policy: detect and report, never silently fix.**
- Unparseable values, rows outside the file's month, and duplicated intervals **raise** an error.
- Missing intervals are **reported** against the full expected 5-minute grid, not filled. Any gap
  handling will be an explicit, documented choice in feature construction.
- Prices are stored exactly as published. **No clipping**: negative prices (to the −$1,000 floor)
  and prices near the market price cap are genuine outcomes, and the spike tail is the target.

**Caching.** One parquet file per month under `data/raw/aemo/VIC1/`, so downloads are resumable
and re-runs are free. Downloads retry with exponential backoff on network errors and 5xx
responses.

**What the data looks like (VIC1, 1 Jan 2023 – 31 Oct 2026, from `nemspike fetch-prices`).**

| | Value |
|---|---|
| Intervals | 403,200 of 403,200 expected (no gaps, no duplicates) |
| Price range | −$1,000.00 to $19,069.69/MWh |
| Negative-price intervals | 88,873 (22 %) |
| Intervals > $300/MWh | 3,165 (0.8 %) |
| Intervals > $1,000/MWh | 313 |
| 99th percentile (full period) | $299.50/MWh |

Talking points:
- About one in five intervals is negative, driven by midday rooftop and utility solar. A model,
  or a target transform, that assumes positive prices would fail immediately.
- The full-period 99th percentile ($299.50) is almost exactly the fixed $300 threshold, so the two
  spike definitions agree on this period. Spikes are rare (under 1 %), so the spike classifier is
  an imbalanced problem: PR-AUC is more informative than ROC-AUC or accuracy.
- The tail is extreme: the maximum is about 350 times the median.

**Licence.** AEMO material is used under AEMO's Copyright Permissions Notice. I could not read the
notice directly (AEMO's website blocks automated clients), so the README carries attribution and a
link to it. **To do before making the repo public:** read the notice in a browser and confirm the
wording.

---

## Limitations (maintained throughout)

- Energy arbitrage only: no FCAS co-optimisation.
- Price taker: no market impact from the battery's own dispatch.
- Simplified battery: constant efficiency, no degradation, no auxiliary load, no ramp limits.
- No network constraints, marginal loss factors or intra-regional congestion.
- Weather from a single Melbourne point, not a demand- or generation-weighted average.
