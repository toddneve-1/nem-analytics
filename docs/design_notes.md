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

## 5. Walk-forward evaluation, baselines and metrics

**Splits.** Expanding-window walk-forward. The first training window is 2023; then 10
development test folds: each quarter of 2024 and 2025, and Q1–Q2 2026. Jul–Oct 2026 is a final
holdout (see below). Each fold re-trains on everything
before it, as a desk would periodically re-train. Fold boundaries are NEM month starts (00:00 AEST).
Never random splits: shuffling would let the model learn from the future.

**Final holdout (Jul–Oct 2026).** All development, model comparison and tuning use the 10
development folds only. The holdout is one extra fold, trained on everything before 1 July 2026,
scored once at the end with `--stage holdout`. Reason: even though no single model sees the
future, *I* see the results of every fold, and repeatedly choosing settings that look best on
those folds makes them optimistic. A period never looked at during development gives an honest
final estimate.

**Embargo.** A training row is used only if its target interval had ended by the start of the
test fold (`target_time <= test_start`). At the 4-hour horizon this removes the last 4 hours of
training origins, whose targets fall inside the test period.

**Spike label.** Price strictly above the threshold ($300/MWh fixed by default). With the
percentile option, the threshold is computed on the **training fold's targets only**.

**Baselines.**
- Point: persistence (latest price), same time yesterday, same time last week, aligned to the
  *target's* time of day.
- Made probabilistic so they can be scored like the quantile models: point + empirical quantiles
  of that baseline's own training-fold errors, estimated **per target hour of day**. Spike
  probability is the training spike rate given whether the reference price was itself a spike.
- Climatology: training-fold quantiles and spike rate by target hour of day, ignoring recent
  history.
All forecasters share one `fit(train, threshold)` / `predict(test)` interface, so baselines and
LightGBM are scored by the same backtest code on exactly the same rows.

**Metrics and why.**
- MAE and RMSE on the point forecast. RMSE is dominated by a few huge spikes; MAE reflects typical
  error. Both are reported.
- Pinball loss per quantile: the proper scoring rule for quantile forecasts (minimised in
  expectation by the true quantile; a test checks this).
- Calibration: share of outcomes below each quantile (ideally q) and q10–q90 coverage (ideally
  80 %).
- Spikes: PR-AUC (average precision), precision and recall at P ≥ 0.5, Brier score. With a base
  rate under 1 %, accuracy and ROC-AUC flatter weak models; PR-AUC is read against the base rate,
  which is what a random classifier scores.

**Baseline results** (`nemspike backtest --models baselines`; 10 development folds, Jan 2024 –
Jun 2026; 262,656 test origins per horizon; spike base rate 0.91 %, 2,402 spike intervals).
Jul–Oct 2026 is the final holdout and is not included.

| Horizon | Model | MAE | RMSE | Mean pinball | q10–q90 coverage | Spike PR-AUC |
|---|---|---|---|---|---|---|
| 30 min | persistence | 24.90 | 234.90 | 8.45 | 0.822 | 0.357 |
| 30 min | same time yesterday | 65.47 | 436.97 | 21.22 | 0.798 | 0.057 |
| 30 min | same time last week | 81.44 | 449.91 | 25.20 | 0.786 | 0.012 |
| 30 min | climatology | 65.62 | 326.31 | 19.86 | 0.793 | 0.027 |
| 4 h | persistence | 72.57 | 446.54 | 20.18 | 0.796 | 0.015 |
| 4 h | same time yesterday | 65.48 | 436.97 | 21.22 | 0.798 | 0.057 |
| 4 h | same time last week | 81.45 | 449.91 | 25.21 | 0.785 | 0.012 |
| 4 h | climatology | 65.63 | 326.32 | 19.86 | 0.793 | 0.027 |

(MAE, RMSE and pinball in $/MWh.)

Talking points:
- At 30 minutes, persistence is hard to beat: prices are sticky, and its spike PR-AUC (0.36) is
  about 39 times the base rate. At 4 hours its edge disappears and spikes are close to
  unpredictable for every baseline. That is where a model must earn its place.
- Pooled coverage looks right (about 80 %), but per fold persistence ranges from 68 % (Q2 2024,
  volatile) to 93 % (Q2 2026, calm). Error distributions learned from the past are too narrow in
  turbulent periods and too wide in calm ones. Volatility is non-stationary, which motivates drift
  monitoring and possibly a shorter or weighted training window.
- Spike counts vary from 661 (Q2 2024) to 15 (Q4 2025) per fold, so per-fold results matter;
  pooled numbers are dominated by the volatile quarters.
- Precision is undefined (NaN) for baselines that never reach P ≥ 0.5: climatology's highest
  hourly spike rate is 0.058. That is why PR-AUC, which needs no threshold, is the headline spike
  metric.

---

## 6. LightGBM quantile models and spike classifier

**Model.** Gradient-boosted trees (LightGBM), one booster per quantile (q0.1, q0.5, q0.9, q0.99)
with the pinball (quantile) loss, plus one binary classifier for P(price > $300/MWh). Separate
models per horizon (direct forecasting). Point forecast = q0.5 (the median).

**Why gradient boosting.** Tabular features with strong interactions (hour × recent volatility ×
temperature), mixed scales, no need to normalise, robust to irrelevant features, fast on
hundreds of thousands of rows, and native quantile and binary objectives.

**Target transform: asinh(price / 100).**
- Near-linear for ordinary prices, log-like for the tail, defined for negative prices (unlike
  log, which fails on the 22 % of negative intervals).
- Quantiles map exactly through any increasing transform, so the back-transformed quantile
  forecasts are unbiased. Means do not, which is another reason the point forecast is the median.
- Effect: the model weights *relative* errors on big prices, rather than letting a single
  $15,000 interval dominate the loss.
- Tested empirically: `lgbm_asinh` against `lgbm_none` (raw prices) on identical folds.

**Fitting procedure (per fold, per quantile and for the classifier).**
1. **Training stride 15 minutes.** Train on every third origin; neighbouring 5-minute rows are
   nearly duplicates. Testing uses every origin.
2. **Early stopping inside the training fold.** The last 60 days of the training fold are the
   validation window, with the same embargo rule: inner-training targets must be realised before
   the window starts. The test fold is never used for any fitting decision.
3. **Refit** on the whole training fold with the chosen number of trees, so the most recent
   60 days, which matter most in a changing market, are not left out of the final model.
4. Fixed hyperparameters (learning rate 0.05, 31 leaves, at least 200 rows per leaf, 80 % feature
   and row sampling, L2 = 1). No hyperparameter search yet; any later tuning uses development
   folds only.

**Quantile crossing.** Separately trained quantile models can cross (for example q0.9 < q0.5 in
some rows). We record the crossing rate and then sort each row. Sorting can only reduce total
pinball loss, so it is a safe repair.

**Spike classifier.**
- No class re-weighting. Re-weighting improves nothing for PR-AUC (a ranking metric) and would
  distort the outputs away from true probabilities, which a trader needs for expected-value
  decisions.
- **Early stopping on log loss, not average precision.** First run: early stopping on average
  precision ran to 1,992 of 2,000 trees, because average precision computed on a 60-day window
  with a handful of spikes is too noisy to stop on. Log loss is smooth and a proper scoring rule.

**Reproducibility.** `seed` from the config, LightGBM `deterministic=True` and
`force_row_wise=True`; a test checks two fits with the same seed give identical predictions.
LightGBM can still differ slightly across machines or thread counts.

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
