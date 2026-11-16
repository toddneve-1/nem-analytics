# Design notes

A running record of design decisions, the reasoning behind them, and known limitations.
Each entry is written to be explainable in an interview: *what* was decided, *why*, and
*what the alternative would have been*.

---

## 0. Market background

**What "demand" is in this data.** `TOTALDEMAND` is the regional demand AEMO used in each 5-minute
dispatch run: the load that scheduled and semi-scheduled generation plus net imports were dispatched
to meet. It is an operational estimate based on real-time measurements, not metered (settlement)
consumption.
- Rooftop solar is behind the meter, so it appears as *lower demand*, not as generation. Sunny
  middays therefore show very low grid demand (down to about 1,200 MW in VIC1).
- Demand is power (MW). Energy in one interval is MW × 5/60 MWh, and value is RRP × MWh.
- Demand for an interval is only known once it is dispatched, so a forecast may use demand up to
  the last completed interval only.

**How the price is set.** Every 5 minutes generators offer quantities at prices. AEMO dispatches the
cheapest offers that meet demand, within network limits, and the marginal offer sets the regional
reference price (RRP) paid to all generators and charged to all consumers in the region.

**Why prices go negative.** It is a market outcome, not a stabilisation payment. When enough
capacity is offered below $0 to meet demand on its own, the clearing price is negative.
Generators offer below zero because:
- renewables earn Large-scale Generation Certificates (LGCs) and often have fixed-price power
  purchase agreements, so they still profit at a modestly negative spot price;
- inflexible thermal plant can find it cheaper to run through a few hours than to shut down and
  restart;
- rooftop solar lowers midday grid demand, so fewer offers are needed and cheap or negative offers
  become marginal.

At a negative price, generators pay to generate and consumers, including a charging battery, are
paid to consume. The market floor is −$1,000/MWh. Price spikes are the opposite case: tight supply
or low wind and solar push dispatch up the offer stack, at times to the market price cap.

**What is out of scope.** Frequency control ancillary services (FCAS), system strength and inertia
are separate markets and arrangements. This project models energy arbitrage only.

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

## 3. Weather loader (Open-Meteo, ERA5)

**Source.** Open-Meteo Historical Weather API, single point (Melbourne CBD; ERA5 snaps it to
the nearest 0.25° grid cell at about −37.75, 145.0), hourly: temperature, humidity, cloud cover,
wind speed and direction at 10 m and 100 m, gusts, and shortwave, direct and diffuse radiation.

**Decisions.**
- **`models=era5` explicitly.** Open-Meteo's default "best match" blends ERA5, ERA5-Land and ECMWF
  IFS, and the blend can change over time. One model keeps the series consistent.
- **`timeformat=unixtime`, `timezone=GMT`.** Unambiguous UTC timestamps. Open-Meteo's ISO format
  applies one fixed offset to the whole response and ignores daylight saving, which is a trap if
  you request Melbourne local time.
- **Variable time conventions.** Temperature, wind and cloud are *instantaneous* at the label.
  Radiation is the *mean over the preceding hour*: the value labelled 12:00 covers 11:00–12:00.
  Both are therefore complete at their label, which we store as `valid_time`. Reading radiation as
  covering the *following* hour would leak an hour of the future; a test pins this down.
- **Rate limits.** API error payloads (for example "daily limit exceeded") are raised, not retried.
  The free tier's limit is per IP address, so a shared cloud IP can be exhausted by other users.
- **Cache validation.** Data is cached per calendar year. Loading checks every expected hour is
  present, so a partial year cached under an earlier, shorter date range fails loudly.

**Data check (1 Jan 2023 – 31 Oct 2026 plus a day of padding).** 33,648 of 33,648 hours, no
missing values, nothing out of physical range.

**Limitations.**
- ERA5 is a **reanalysis published about 5 days after real time**. Here it stands in for real-time
  observations a trader would have had. It is also a 25 km grid average, smoother than a station.
- One point for the whole of Victoria. Wind farms are mostly in western Victoria, solar in the
  north-west, and demand is concentrated in Melbourne.

---

## 4. Feature construction and leakage control

**Framing.** Each row is a forecast origin `T`: the end of the latest completed 5-minute
interval. The target is the RRP of the interval ending at `T + h`. One table per horizon (direct
multi-horizon forecasting, rather than feeding forecasts back in recursively, which compounds
errors).

**What a feature may use at origin `T`.**

| Family | Examples | Why it is allowed |
|---|---|---|
| Price/demand history | lags 0–48 intervals; 1 h, 4 h and 24 h rolling mean, std, min, max, median, share negative, count above $300; recent changes | Intervals ending at or before `T` are published |
| Seasonal | price and demand at the target's time of day 1 day and 1 week earlier | `T + h − 24 h ≤ T` because `h ≤ 24 h` (enforced) |
| Calendar (target time) | hour, minute of day, weekday, month, VIC public holiday | Known in advance |
| Solar geometry (target time) | sun elevation at the target interval midpoint | Astronomy, known in advance; a leak-free stand-in for "how sunny could it be" |
| Weather (pluggable) | latest temperature, wind (u/v at 10 m and 100 m), cloud, radiation, 3 h changes, 24 h max temperature, data age | Joined *as of* `valid_time + 1 h ≤ T` |

**Key choices.**
- **Regular 5-minute grid.** Features are computed after reindexing to a complete grid, so
  `shift(k)` always means *k intervals earlier*. Gaps stay NaN; nothing is forward-filled.
- **Calendar uses the target interval's start in NEM time,** so the interval 23:55–24:00 belongs
  to the day it falls in. Victorian holidays include Melbourne Cup Day and the Friday before the
  AFL Grand Final, which a national calendar would miss.
- **Wind direction as u/v components.** Degrees are circular (359° is next to 1°), so raw degrees
  mislead a model. Speed and direction become eastward and northward components.
- **Pluggable weather.** Builders subclass `WeatherFeatureBuilder`, register by name, and are
  selected in the config (`features.weather_builders`). They receive weather already tagged with
  `available_at` and can attach it to origins only through `asof_join`. That is where wind-flow
  and topography features will go.
- **Targets stay raw.** Transforms (asinh) and spike labels are applied inside each training fold
  later, so any fitted threshold only sees training data.
- **Warm-up.** The first week is dropped because weekly lags are undefined.

**How leakage is tested.**
1. **Unit tests** pin each alignment: target = price at `T + h`; lag 0 = price at `T`; rolling
   windows end at `T`; seasonal lags point at the target's time of day; radiation for the hour
   ending 12:00 is first usable at 13:00 with a 1 h lag.
2. **Future-perturbation test (end to end).** For sampled origins, replace every price, demand and
   weather value that becomes known after `T` with random noise, rebuild all features, and require
   the row at `T` to be identical. This covers every feature at once, including future ones.
3. **Testing the test.** A deliberately leaky feature (next interval's price) and a weather build
   that ignores the lag must both be caught.
4. The same check passes on a month of **real** data with weather.

**Size.** 401,184 origins × 74 features per horizon (15 from weather), built in about 1.5 s.

---

## Planned extensions (after the core pipeline)

1. **AEMO pre-dispatch benchmark.** Compare the models against AEMO's own published forecasts:
   5-minute pre-dispatch (P5MIN, next hour) for the 30-minute horizon and 30-minute pre-dispatch
   (PREDISPATCH) for the 4-hour horizon. For each forecast origin, use only the most recent
   pre-dispatch run *published before* that origin. This is a much stronger benchmark than
   persistence, because pre-dispatch reflects participants' current bids. Pre-dispatch may then be
   tried as a feature with the same as-of rule. To check: file sizes in the NEMWEB MMSDM archive
   and whether NEMOSIS supports the required tables.
2. **Melbourne Airport station observations** (NOAA Integrated Surface Database, METAR reports)
   as a second, pluggable weather source: real-time-observable data versus ERA5 reanalysis.
   Confirm WMO redistribution terms for non-US stations first.
3. **Archived weather forecasts** (Open-Meteo historical-forecast API) for the 4-hour horizon,
   with care over forecast issue times.

---

## Limitations (maintained throughout)

- Energy arbitrage only: no FCAS co-optimisation.
- Price taker: no market impact from the battery's own dispatch.
- Simplified battery: constant efficiency, no degradation, no auxiliary load, no ramp limits.
- No network constraints, marginal loss factors or intra-regional congestion.
- Weather from a single Melbourne point, not a demand- or generation-weighted average.
