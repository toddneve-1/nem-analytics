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
  outside its limits) fails at load time rather than silently changing the experiment. Unknown
  keys are rejected too (`extra="forbid"`): otherwise a misspelt optional key would quietly fall
  back to its default. Every key is read by the code; the dispatch grid, re-plan interval and
  look-ahead of the forecast-driven battery come from the config, not from defaults in the code.
- Raw data, MLflow runs and model artefacts are git-ignored: they are reproducible from code,
  and committing data invites stale or hand-edited inputs.

**CI.** GitHub Actions runs `ruff check`, `ruff format --check` and `pytest` on Python 3.11 and
3.12. Tests use small synthetic fixtures; any test that touches the network is marked
`network` and skipped in CI, so CI is deterministic and does not depend on AEMO's servers. An
end-to-end test runs `backtest`, `battery` and `battery-forecast` through the command line on
three months of synthetic data, so the README's command sequence and the fold wiring are tested.

**Default experiment settings (agreed up front).**

| Setting | Value | Note |
|---|---|---|
| Region | VIC1 | |
| Period | 2023-01-01 to 2026-10-31 | Full months, post 5-minute settlement |
| Horizons | 30 min, 4 h | Plus a 30-minute grid to 4 h for dispatch |
| Quantiles | 0.1, 0.5, 0.9, 0.99 | |
| Spike | > $300/MWh (fixed) | Percentile option, fitted on training fold only |
| Battery | 100 MW / 200 MWh, 90 % round trip, SoC 10–90 % | Continuous operation; wear cost $40/MWh discharged (Step 8; $0–60 sensitivity) |
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

**Results** (`nemspike backtest --models lgbm_asinh,lgbm_none`; same 10 development folds and
262,656 test origins per horizon as the baselines; seed 42).

| Horizon | Model | MAE | RMSE | Mean pinball | Pinball q0.99 | q10–q90 coverage | Spike PR-AUC | Precision / recall at 0.5 |
|---|---|---|---|---|---|---|---|---|
| 30 min | persistence (best baseline) | **24.90** | **234.90** | **8.45** | **4.20** | 0.822 | 0.357 | 0.61 / 0.41 |
| 30 min | lgbm_asinh | 25.47 | 311.07 | 8.55 | 4.92 | 0.787 | **0.594** | 0.73 / 0.45 |
| 30 min | lgbm_none | 25.51 | 311.22 | 8.88 | 5.69 | 0.795 | 0.594 | 0.73 / 0.45 |
| 4 h | best baseline *per column*¹ | 65.48 | 326.32 | 19.86 | 10.16 | 0.793 | 0.057 | – |
| 4 h | lgbm_asinh | **41.17** | **315.99** | **13.57** | **9.40** | 0.733 | **0.269** | 0.51 / 0.06 |
| 4 h | lgbm_none | 41.17 | 315.88 | 13.74 | 9.70 | 0.730 | 0.269 | 0.51 / 0.06 |

(Spike base rate 0.91 %. MAE, RMSE and pinball in $/MWh.) ¹ Each 4 h baseline figure is the best of
the four baselines on that metric: MAE same time yesterday; RMSE, mean pinball and coverage
climatology; q0.99 pinball persistence; PR-AUC same time yesterday. No single baseline achieves
all of them.

Per fold (`reports/metrics/dev/h*_by_fold.csv`):
- **4 h:** LightGBM beats the best baseline on MAE and on mean pinball in **10 of 10** quarters.
  Pooled: MAE −37 % against same time yesterday, mean pinball −32 % against climatology.
- **30 min:** LightGBM beats persistence in **7 of 10** quarters on both MAE and pinball, but
  loses on the pooled totals. Two volatile quarters (Q1 2024, Q2 2025) dominate: in Q2 2025 its
  MAE is $58.8 against persistence's $43.8.
- **Spikes:** best PR-AUC in every quarter at both horizons. At 30 min, 0.594 is about 65 times
  the base rate (persistence 0.357); at 4 h, 0.269 is about 30 times the base rate (best
  baseline 0.057).

Interpretation and talking points:
- **The model earns its keep at 4 hours,** exactly where the baselines fail. At 30 minutes the
  latest price already carries most of the information, and persistence is a hard benchmark.
  This is a common finding in short-term electricity price forecasting.
- **The 30-minute loss on pooled MAE is concentrated in volatile quarters.** A plausible cause is
  that the model predicts the *level* of the price; when prices jump, a median forecast that
  partly reverts towards typical levels misses by more than "stay where you are". A natural next
  experiment, run on development folds only: predict the *change* from the current price, or add
  persistence as an explicit fallback.
- **The 4-hour intervals are too narrow:** q10–q90 coverage 73 % against a nominal 80 %, and
  under 70 % in the volatile 2024 quarters. Quantile models trained on a growing window
  understate uncertainty when volatility rises. A candidate fix: conformal recalibration of the
  intervals using recent residuals, also tested on development folds only.
- **asinh against raw prices:** identical MAE, but asinh is better on every quantile loss,
  especially the tail: q0.99 pinball 4.92 against 5.69 at 30 min (−14 %) and 9.40 against 9.70
  at 4 h (−3 %). The transform helps where it should: the heavy tail. `lgbm_asinh` is the
  default model.
- **Identical spike metrics for both transforms are a sanity check, not a coincidence:** the
  classifier does not use the price transform and is trained with the same seed, so it is the
  same model.
- **At 4 h the classifier ranks well (PR-AUC 0.27) but rarely reaches P ≥ 0.5** (recall 6 %).
  Spikes 4 hours ahead are inherently uncertain, so the model's probabilities are honest and
  low. A trader would act on a lower threshold, for example "P > 0.1", chosen from the cost of
  false alarms against missed spikes. Precision and recall at 0.5 are therefore not the right
  operating point, which is why PR-AUC is the headline metric.

**All horizons, 30 min to 4 h in 30-minute steps** (`nemspike backtest --horizons
60,90,120,150,180,210`). These intermediate horizons were run for Step 9, which needs a forecast
price path over the battery's 4-hour planning window. Same folds, same settings.

| Horizon | LightGBM MAE | Best baseline MAE | LightGBM mean pinball | Best baseline pinball | q10–q90 coverage | LightGBM PR-AUC | Best baseline PR-AUC | Quarters LightGBM wins on MAE |
|---|---|---|---|---|---|---|---|---|
| 30 min | 25.47 | 24.90 (persistence) | 8.55 | 8.45 | 0.787 | 0.594 | 0.357 | 7 / 10 |
| 1 h | 29.78 | 35.15 (persistence) | 10.22 | 11.53 | 0.769 | 0.398 | 0.232 | 10 / 10 |
| 1.5 h | 32.70 | 43.95 (persistence) | 11.31 | 13.95 | 0.759 | 0.377 | 0.135 | 10 / 10 |
| 2 h | 35.06 | 51.84 (persistence) | 11.91 | 16.02 | 0.748 | 0.337 | 0.087 | 10 / 10 |
| 2.5 h | 37.05 | 57.97 (persistence) | 12.47 | 17.33 | 0.739 | 0.318 | 0.064 | 10 / 10 |
| 3 h | 38.59 | 63.60 (persistence) | 12.86 | 18.53 | 0.737 | 0.274 | 0.057 | 10 / 10 |
| 3.5 h | 40.02 | 65.48 (same time yesterday) | 13.20 | 19.45 | 0.736 | 0.287 | 0.057 | 10 / 10 |
| 4 h | 41.17 | 65.48 (same time yesterday) | 13.57 | 19.86 | 0.733 | 0.269 | 0.057 | 10 / 10 |

(Best baseline per column; pinball baseline is the best of the four at that horizon.)

Talking points:
- **The crossover is between 30 and 60 minutes.** Persistence's error grows fast with horizon
  (from $25 to $64 by 3 hours); the model's grows slowly ($25 to $41 at 4 hours). From 1 hour on,
  the model wins every quarter at every horizon.
- **Errors and coverage degrade smoothly with horizon**, as they should; a jump would suggest a
  bug. Coverage drifts from 79 % to 73 %, so interval recalibration matters more at longer
  horizons.
- PR-AUC falls from 0.59 to about 0.27, with a small, non-monotone bump at 3.5 h (0.287 against
  0.274 at 3 h): within the noise of a metric driven by about 2,400 spike intervals.

---

## 7. Experiment tracking and versioned models (MLflow, local)

**What is tracked.** Every `nemspike backtest` run logs one MLflow run per (stage, horizon,
model):
- the full config, flattened to parameters (for example `model.learning_rate`, `seed`);
- tags: stage (`dev` or `holdout`), horizon, model, **git commit** and whether the working tree
  had uncommitted changes;
- pooled metrics (MAE, pinball per quantile, coverage, PR-AUC and so on);
- per-fold metrics as a series (step = fold number), plus LightGBM diagnostics per fold
  (quantile crossing rate, early-stopping rounds);
- the config file and per-fold table as artefacts.

So any number in the README can be traced to a run, a commit, a config and a seed.

**Versioned models.** `nemspike train-final` fits a model on all data up to a cut-off (by default
the holdout start) and registers it in the MLflow **Model Registry** as a new version
(`nemspike-vic1-lgbm_asinh-h30` version 1, 2, ...). Each version is an MLflow `pyfunc` model
wrapping the fitted forecaster: `predict(features)` returns the quantiles, the point forecast and
the spike probability. Versions are tagged with their training window, spike threshold and git
commit. Package versions are pinned explicitly from the environment rather than inferred.

**Storage.** Local SQLite (`mlflow.db`) for runs and the registry, `mlartifacts/` for files;
both git-ignored. Browse with `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

**Why local MLflow.** It is the common open-source standard, needs no server for one person, and
the same code points at a shared tracking server by changing one URI in the config.

**Reproducibility check.** All backtests were re-run under MLflow tracking (12 runs: six models,
two horizons) and compared with the earlier untracked run: all 12 prediction files were
**bit-for-bit identical**, and so were the summary tables. Same code, data and seed give the same
numbers on this machine.

**Registered models.** `nemspike train-final --models lgbm_asinh` registered
`nemspike-vic1-lgbm_asinh-h30` and `-h240`, version 1, each trained on about 366,000 origins with
targets up to 1 July 2026 00:00 AEST (the holdout start), so they are the models to be scored once
on the holdout. A smoke test loads a version from the registry and predicts quantiles, point
forecast and spike probability.

**LightGBM diagnostics (development folds, `reports/metrics/dev/h*_lgbm_asinh_diagnostics.csv`).**
- Quantile crossing before sorting: 0.4–1.6 % of rows at 30 min and 0.3–2.1 % at 4 h; sorting
  repairs it.
- Early stopping chose a median of 419–544 trees for the median models, but the spike classifier
  ran to 1,872–1,992 trees (near the 2,000 cap) in its worst folds. Candidate refinements, to be
  tested on development folds only: a higher learning rate for the classifier, or a larger
  early-stopping patience trade-off.

**Operational lesson.** Two LightGBM jobs on the same 4-core machine slowed each other by an
order of magnitude (a 40-second fold took over 10 minutes), because LightGBM's threads spin
while waiting for CPU. The thread count is now configurable (`model.num_threads`) and the unit
tests use one thread.

---

## 8. Battery optimiser and benchmark strategies

**Formulation (over a planning period of N five-minute intervals, dt = 1/12 h).**

    maximise   Σ_t [p_t (d_t − c_t) − k d_t] dt                   (revenue net of wear)
    subject to s_t = s_{t−1} + η c_t dt − d_t dt / η                (energy balance)
               0 ≤ c_t, d_t ≤ 100 MW                              (power)
               20 ≤ s_t ≤ 180 MWh                                 (10–90 % of 200 MWh)
               s_0 = s_N = 100 MWh                                (start and end the period at 50 %)

with η = √0.9 on each leg (90 % round trip) and a wear cost k per MWh discharged (added later; see
below). Battery parameters come from the config.

**Planning period.** First implemented per NEM day (N = 288: every day starts and ends at 50 %,
so days are independent). Now continuous by default: N is a whole test quarter (about 26,000
intervals), with 50 % only at the quarter's start and end. See "Continuous operation" below.

**The negative-price problem.** The plain LP, at negative prices with a full battery, charges and
discharges *at the same time*: the round-trip losses turn the battery into a load that is paid to
consume. A real battery has one inverter and cannot do this. On 40 sampled real days the LP did
this in 909 intervals and overstated revenue by 0.45 % (up to $2,336 on one day).

Forbidding it makes the problem non-convex. Approaches tried:

| Approach | Result on 40 sampled days |
|---|---|
| Plain LP | 0.4 s, but physically impossible dispatch; an upper bound |
| MILP, binary on every negative-price interval | Exact, but one day took 292 s (days with ~200 negative intervals) |
| LP, then "repair" the overlap | Fast, but 0.5 % below exact |
| Iteratively forbid discharge where the LP overlaps | Fast, 0.8 % below exact |
| Lazy MILP (binaries only where the LP overlaps) | Exact, but still up to 200 s on hard days |
| Tighter MIP gap and price tie-breaking | No material speed-up on hard days |
| **Hybrid: LP, with dynamic programme fallback** | **0.7 s for 40 days, within 0.02 % of exact, never overlaps** |

**Chosen method (`method="hybrid"`).**
1. Solve the LP. If it has no simultaneous charge and discharge at a negative price, it is
   provably optimal for the real (exclusive) problem. Any overlap at prices ≥ 0 is removed by
   reducing charge by x and discharge by η²x, which leaves the state of charge unchanged and
   cannot lower revenue.
2. Otherwise solve a **dynamic programme**: state of charge on a grid of 0.44 MWh steps,
   backward induction over the 288 intervals, one action per interval (charge, discharge or idle,
   so overlap is impossible by construction). The grid is anchored on the starting state of
   charge and sized so that full charge power (18 steps) and full discharge power
   (20 steps) are exactly representable at 90 % round-trip efficiency.

The exact MILP is kept for validation (tests compare the hybrid against it), and the plain LP is
kept as an upper bound.

**Strategies.**
- **Perfect foresight:** optimise each day on its actual prices. Upper bound.
- **Naive time-of-day:** average price for each 5-minute slot of the day over the training
  period (all data before the test fold), optimise once on that average day, then repeat the
  fixed schedule every test day. No forecasts, no test-period information; re-fitted for each
  fold. It can lose money: averaging prices by slot picks up spikes that happened to fall in
  particular slots, and the schedule then "chases" spikes that do not recur.
- Forecast-driven rolling-horizon dispatch: Step 9.

**First results, no wear cost** (`nemspike battery --degradation-costs 0`; 912 NEM days,
1 Jan 2024 – 30 Jun 2026, development folds only; 100 MW / 200 MWh, 90 % round trip, 10–90 % state
of charge). The wear-cost results that supersede these as the headline are further below.

| Strategy | Revenue | Per day | $/MW-year | Capture vs perfect | Loss-making days | Cycles per day |
|---|---|---|---|---|---|---|
| Perfect foresight | $52.69 m | $57,778 | $210,888 | 100 % | 0 % | 4.87 |
| Naive time-of-day | $22.04 m | $24,171 | $88,224 | **41.8 %** | 7 % | 3.37 |

(Cycles = energy discharged ÷ 160 MWh usable capacity.)

Checks on the saved dispatch, all 912 days, repeated for wear costs of $0, $10 and $20: state of
charge within 20–180 MWh, power ≤ 100 MW, no simultaneous charge and discharge, every day ends at
100 MWh, energy balance exact to 1e-13 MWh, and the naive strategy never beats perfect foresight
(on net revenue) on any day.

Talking points:
- **Revenue is concentrated in spikes.** The 5 best days earn 23 % of all perfect-foresight
  revenue; the median day earns $40,619 and the 99th-percentile day $254,339. Missing a handful
  of spike days costs more than months of ordinary trading, which is why spike forecasting is
  the commercially relevant problem.
- **A fixed daily schedule captures about 42 %.** "Charge at midday solar, discharge in the
  evening peak" is worth something, but it misses the days that matter and loses money on 7 % of
  days.
- **Perfect foresight cycles 4.9 times a day**, far above a typical real battery (often around
  1–2). With no degradation cost, the optimiser trades every 5-minute wiggle that covers the 10 %
  round-trip loss. The $/MW-year figure is therefore an optimistic upper bound for energy
  arbitrage alone, not a forecast of real earnings. This led to adding a degradation cost (next).

**Degradation cost (added after the first results).** With no wear cost, perfect foresight cycled
4.9 times a day, trading every 5-minute wiggle that covered the 10 % round-trip loss. A linear wear
cost of k $/MWh discharged is now in the objective (LP and dynamic programme): a trade must cover
losses *and* the battery life it uses. Real wear costs depend on cell chemistry, warranty terms
and replacement prices, so instead of claiming one "true" value, $10/MWh is used as an
illustrative central case with $0 and $20 as sensitivities (`nemspike battery
--degradation-costs 0,10,20`). Capture is measured on **net** revenue (market revenue minus wear).
These results use the **daily** rule set; they are superseded as the headline by the continuous,
$40/MWh results in the next section, and are kept for comparison.

| Wear cost | Strategy | Market revenue | Wear cost | Net revenue | Net $/MW-year | Capture | Loss-making days | Cycles/day |
|---|---|---|---|---|---|---|---|---|
| $0 | Perfect foresight | $52.69 m | – | $52.69 m | $210,888 | 100 % | 0 % | 4.87 |
| $0 | Naive time-of-day | $22.04 m | – | $22.04 m | $88,224 | 41.8 % | 7.0 % | 3.37 |
| $10 | Perfect foresight | $51.66 m | $4.67 m | $46.99 m | $188,047 | 100 % | 0 % | 3.20 |
| $10 | Naive time-of-day | $22.29 m | $3.54 m | $18.75 m | $75,037 | 39.9 % | 14.7 % | 2.42 |
| $20 | Perfect foresight | $50.12 m | $7.19 m | $42.93 m | $171,795 | 100 % | 0 % | 2.46 |
| $20 | Naive time-of-day | $21.50 m | $5.23 m | $16.28 m | $65,153 | 37.9 % | 21.3 % | 1.79 |

The $0 row reproduces the earlier run exactly. Observations:
- Wear cost cuts cycling sharply (4.9 → 3.2 → 2.5 per day for perfect foresight) but net revenue
  much less (−11 % at $10): the trades it removes were thin.
- The naive schedule's *market* revenue slightly rises at $10, because it stops making marginal
  trades that often lost money once realised prices differed from the average day.
- Revenue concentration at $10: the 5 best days earn 25.8 % of perfect-foresight net revenue and
  the best 10 % of days 42.8 % (median day $34,088; 99th percentile $247,835).
- Even at $20, perfect foresight cycles about 2.5 times a day, still high: with perfect
  knowledge of 5-minute prices there are many profitable intraday spreads. Forecast-driven
  strategies will cycle less, because uncertain spreads are not worth trading.

**Continuous operation and the wear-cost sweep (current headline).**

*Continuous instead of daily.* The daily reset was chosen for a clean comparison and per-day
results, not for computation: a whole quarter (about 26,000 intervals) solves in about 5 seconds.
What makes comparisons fair is that every strategy follows the *same* rule set and differs only
in information. Continuous operation is the more realistic rule set (no operator empties to 50 %
every midnight), so all strategies now run continuously: state of charge is carried across
midnights and fixed at 50 % only at the start and end of each test quarter. The naive schedule
returns to 50 % each night anyway, so it is valid under both rule sets and unchanged. The daily
rule set is kept as a comparison (`--mode daily`).

*Naive schedule fix (after review).* The naive schedule was first built under the daily rule
(50 % at midnight) even when evaluated continuously. Inspecting the Q1 2026 schedule showed the
consequence: after the evening peak it refilled at about $81/MWh between 22:00 and 24:00 and sold
that energy at about $88/MWh at 06:00–07:00, which loses money after round-trip losses ($88 × 0.9 ≈
$79) even before wear. The trade was forced by the midnight rule, not chosen by the prices. Under
continuous rules the naive schedule is now a **repeating daily cycle**: each day ends where it
started, and the optimiser chooses that overnight level (`optimise(..., cyclic=True)`; if the
cyclic LP overlaps at a negative price, the exact MILP is used, since these are single-day
problems). The Q1 2026 schedule became one clean cycle (charge 11:25–13:15 at about $1/MWh,
discharge 17:25–19:10 at about $170–180/MWh, overnight at minimum charge), and naive capture at
$40/MWh rose from 33.9 % to 42.0 % (loss-making days 29.7 % → 16.7 %). Over a quarter the naive
battery starts and ends at its own overnight level, so it gains no net energy, which keeps it
comparable with perfect foresight's start and end at 50 %. Lesson: inspect what a benchmark
actually does, not only its score; a weak benchmark flatters any model compared against it.

*Wear cost of $40/MWh.* Derived as cost of replacing worn capacity ÷ lifetime energy delivered,
with illustrative assumptions rather than market data: $200,000 per MWh of cells ÷ (6,000 cycles ×
80 % usable depth) ≈ $40/MWh. Plausible inputs span roughly $20–60/MWh, so the sweep covers $0–60.

| Wear cost | Strategy | Net revenue | Net $/MW-year | Capture vs perfect | Loss-making days | Cycles/day |
|---|---|---|---|---|---|---|
| $0 | Perfect foresight | $53.47 m | $213,995 | 100 % | 0.0 % | 4.80 |
| $0 | Naive time-of-day | $22.97 m | $91,911 | 43.0 % | 7.2 % | 3.10 |
| $10 | Perfect foresight | $47.90 m | $191,724 | 100 % | 0.0 % | 3.10 |
| $10 | Naive time-of-day | $20.18 m | $80,780 | 42.1 % | 12.9 % | 2.13 |
| $20 | Perfect foresight | $43.99 m | $176,052 | 100 % | 0.1 % | 2.37 |
| $20 | Naive time-of-day | $17.64 m | $70,596 | 40.1 % | 17.7 % | 1.71 |
| **$40** | **Perfect foresight** | **$38.21 m** | **$152,938** | **100 %** | **0.2 %** | **1.68** |
| **$40** | **Naive time-of-day** | **$16.06 m** | **$64,273** | **42.0 %** | **16.7 %** | **0.96** |
| $60 | Perfect foresight | $33.91 m | $135,717 | 100 % | 0.9 % | 1.31 |
| $60 | Naive time-of-day | $13.43 m | $53,744 | 39.6 % | 24.2 % | 0.95 |

(912 development days, Jan 2024 – Jun 2026, continuous rule set. Net = market revenue minus wear
cost. `nemspike battery --mode continuous --degradation-costs 0,10,20,40,60`.)

Checks on all continuous dispatch, every wear cost: state of charge within 20–180 MWh, power ≤
100 MW, no simultaneous charge and discharge, every quarter ends at 100 MWh, energy balance exact to
about 1e-13 MWh. At $40/MWh the whole-quarter LP never uses the negative-price loophole (burning
energy no longer pays), so the LP solution is provably optimal.

Talking points:
- **Continuous beats daily by 1.5–4.7 %, and the gain grows with wear cost** (+1.5 % at $0,
  +3.6 % at $40, +4.7 % at $60). With costly cycles, the optimiser does fewer, larger trades, so
  the option to hold energy overnight for a better opportunity is worth relatively more.
- **At $40/MWh perfect foresight cycles 1.7 times a day**, in the range of real operation, against
  4.8 with no wear cost. Net revenue falls 29 % from $0 to $40: the trades removed were the thin
  ones.
- **The naive schedule captures about 40–43 % at every wear cost** and still loses money on 17 %
  of days at $40: a fixed schedule cannot skip days when the spread is too small to pay for wear.
  This is where forecasts should add the most value.
- **Naive capture is not monotone in wear cost** (42.1 % at $10, 40.1 % at $20, 42.0 % at $40).
  The average-day schedule switches from about 1.7 to about 1 cycle a day between $20 and $40:
  it drops a morning cycle that looks profitable on the *average* day but often lost money on real
  days. A schedule optimised on averages can choose trades whose realised value is worse.
- **Value is even more concentrated with realistic wear cost:** at $40 the 5 best days earn 31.7 %
  of perfect-foresight net revenue and the best 10 % of days 50.1 %.
- A few perfect-foresight days are slightly negative under continuous operation (0.2 % at $40).
  That is expected, not an error: the battery may buy on one day to sell on the next.

**Assumptions and limitations.** Price taker (no market impact), energy only (no FCAS), a
linear wear cost (real degradation depends on depth of cycle, temperature and age), constant efficiency, no auxiliary load, no
ramp limits, no network constraints or marginal loss factors, settlement at the 5-minute RRP.

---

## 9. Forecast-driven battery (in progress)

### 9.1 Valuing energy left at the end of a planning window: grouping analysis

The forecast-driven battery plans 4 hours ahead (the forecasts' reach), re-planning every 30
minutes. Each plan needs a value for energy still stored at the window's end; otherwise it treats
leftover energy as worthless, will not charge for a peak just beyond the window, and sells too
early. The chosen approach (option C) is an empirical **water-value curve**: the marginal value of
stored energy from perfect foresight on training data, averaged by time of day.

**What the marginal value is.** The shadow price (dual) of the energy-balance constraint in the
perfect-foresight LP: how much one extra MWh in storage at that interval would add to revenue.
`nemspike.battery.water_value.storage_values` computes it. Simultaneous charge and discharge at
negative prices appeared in 16 intervals across three quarters even at $40/MWh wear (all below
−$370/MWh); for those, the charge-or-discharge choice is fixed from the exact solution and the LP
re-solved, so the duals are exact. Tests check the dual against a finite difference and two
textbook cases: when charging is easy before a peak, stored energy is worth its **replacement
cost** (cheap price ÷ charging efficiency); when charging is constrained, it is worth the **peak
price** after discharge losses and wear. The water value reflects whichever constraint binds.

**Should the curve depend on day type? Tested, not assumed** (`uv run python
scripts/analysis/water_value_grouping.py`). Values were computed for 2023–2025 (each quarter
optimised continuously at $40/MWh). Curves were learned on 2023–2024 and scored on 2025:

| Grouping | Groups | 2025 MAE ($/MWh) | 2025 RMSE ($/MWh) |
|---|---|---|---|
| Time of day only | 288 | 58.44 | 277.47 |
| **Time of day × weekday/weekend (public holidays as weekend)** | 576 | **57.86** | **277.00** |
| Time of day × day of week | 2,016 | 59.28 | 277.98 |

(Mean curves; median curves rank the same way. Output in `reports/metrics/dev/water_value_grouping.csv`.)

Findings:
- **Weekends are genuinely different:** average storage value $26–32/MWh on Saturday and Sunday
  against $49–68/MWh on weekdays. At weekend middays it is even negative (−$17 to −$31/MWh at
  13:00): stored energy is a liability when you would rather have empty room to be paid to charge.
- **Day-of-week differences do not generalise.** Apparent weekday differences (Thursday 18:00
  $224 against $91–129 on other weekdays; Tuesday 13:00 $88 against $1–9) come from a handful of
  spike days in 2023–2024 and make 2025 predictions worse: seven groups leave too few days each to
  average away spikes (overfitting).
- **Decision:** time of day × weekday/weekend, public holidays treated as weekends, using the
  **mean** (the expected value is the economically relevant quantity, and the mean is best on
  RMSE, which it is designed to minimise).
- **Caveat:** errors around $58/MWh show that storage value swings a lot from day to day. The
  curve captures the typical value, not any particular day, which is a limitation of option C.

### 9.2 Turning quantile forecasts into an expected price

The battery should plan on the **expected** price (revenue is linear in price, so a risk-neutral
operator maximises expected revenue). The expected price is the probability-weighted average of
the price distribution, equivalently the average of the quantile function over all probability
levels. With only q0.1, q0.5, q0.9 and q0.99, the position of the mass within each band and the
tails beyond q0.1 and q0.99 are unknown, so four constructions were compared against realised
prices (`uv run python scripts/analysis/expected_price_methods.py`; development folds 1–9, since
methods c and d learn only from *earlier* folds; 236,448 forecasts per horizon):

- **a, linear:** each band's mean is its midpoint; tails taken at q0.1 and q0.99.
- **b, asinh:** quantile function linear on the asinh scale between quantiles, tails extrapolated;
  band means in closed form.
- **c, learned:** from earlier folds, the empirical probability of each band (correcting the
  too-narrow intervals), the average position of outcomes inside each band, and the average excess
  beyond q0.1 and q0.99.
- **d, hurdle:** (1 − P(spike)) × mean of the quantiles capped at $300 + P(spike) × average
  historical spike size.

| Horizon | Median (q0.5) RMSE / bias | a RMSE / bias | b RMSE / bias | c RMSE / bias | d RMSE / bias |
|---|---|---|---|---|---|
| 30 min | 283.1 / −9.2 | **272.1** / −2.8 | 273.9 / −4.0 | 273.1 / +1.7 | 298.7 / +9.8 |
| 1 h | 284.7 / −9.8 | **281.0** / −4.3 | 281.4 / −5.5 | 281.2 / +2.4 | 302.4 / +10.1 |
| 2 h | 286.0 / −9.9 | **284.8** / −4.7 | 285.0 / −5.8 | 284.9 / +2.4 | 297.1 / +9.1 |
| 3 h | 286.9 / −10.6 | 285.9 / −5.0 | 286.2 / −6.3 | **285.8** / +3.3 | 294.5 / +7.3 |
| 4 h | 287.8 / −11.8 | **286.8** / −5.2 | 287.1 / −6.7 | 286.9 / +2.4 | 291.9 / +5.9 |

($/MWh. Bias = average of forecast minus actual. Full table, all eight horizons, with calibration
slope and top-decile checks: `reports/metrics/dev/expected_price_methods.csv`.)

Findings:
- **a, b and c are practically tied on RMSE** (within about 0.6 % of each other at every horizon)
  and all beat the median, which **underestimates the mean by $9–12/MWh** (as expected for a
  right-skewed distribution: the median ignores the spike tail).
- **c is the least biased** (+$1.7 to +$3.3) because it learns from realised outcomes; a and b
  remain slightly low (−$3 to −$7).
- **d (hurdle) is clearly worst**, worse than the median on RMSE, with a calibration slope of
  0.5–0.8: when the spike probability is high it multiplies by the average historical spike size
  and overshoots. The classifier ranks spike risk well (PR-AUC) but its probability times a fixed
  spike size is not a good estimate of the expected price.
- **All methods under-predict high prices:** calibration slopes of 1.3–1.5 for a, b and c, and in
  the top decile of forecasts the realised average is about $283–305/MWh against forecasts of
  about $200–257. The quantile forecasts carry limited information about spike *magnitude*.
- **RMSE barely improves on the median** (about 4 % at 30 minutes, under 1 % at 4 hours). RMSE is
  dominated by a few extreme intervals that no method predicts.

### 9.3 First forecast-driven results (water value, $40/MWh wear, continuous)

`nemspike battery-forecast`: rolling 4-hour plans every 30 minutes; leftover energy valued with
the weekday/weekend water-value curve learned from earlier data; development folds 1–9 (Apr 2024 –
Jun 2026, 821 days; fold 0 is excluded because the learned planning price needs earlier forecasts).
Benchmarks are evaluated on the same days.

| Strategy | Net revenue | Net $/MW-year | Capture vs perfect | Loss-making days | Cycles/day |
|---|---|---|---|---|---|
| Perfect foresight | $33.06 m | $146,994 | 100 % | 0.2 % | 1.69 |
| Forecast, median price | $15.76 m | $70,066 | 47.7 % | 8.4 % | 1.17 |
| Forecast, linear expected price (a) | $15.76 m | $70,044 | 47.7 % | 9.0 % | 1.16 |
| **Forecast, learned expected price (c)** | **$15.56 m** | **$69,161** | **47.0 %** | **9.4 %** | **1.14** |
| Naive time-of-day | $15.21 m | $67,600 | 46.0 % | 17.7 % | 0.96 |

Capture by quarter (net revenue as % of perfect foresight):

| Quarter | Forecast (c) | Forecast (median) | Naive |
|---|---|---|---|
| 2024 Q2 | 49.4 | 50.5 | 41.1 |
| 2024 Q3 | 51.9 | 50.2 | 47.2 |
| 2024 Q4 | 53.3 | 51.4 | 37.9 |
| 2025 Q1 | 50.7 | 51.4 | 33.9 |
| **2025 Q2** | **36.1** | **38.3** | **59.2** |
| 2025 Q3 | 56.4 | 60.0 | 54.8 |
| 2025 Q4 | 50.1 | 53.1 | 35.8 |
| 2026 Q1 | 44.6 | 42.0 | 34.3 |
| 2026 Q2 | 25.7 | 30.3 | 12.7 |

Checks: physically valid dispatch for all planning prices (energy balance exact, limits respected,
no simultaneous charge and discharge, every quarter ends at 50 %).

**How certain are these results?** Revenue is concentrated in a few days, so every headline is
reported with a 95 % interval from a paired block bootstrap
(`nemspike.battery.strategies.paired_bootstrap` and `capture_bootstrap`): the same days for all
strategies, resampled in blocks of 7 consecutive days (10,000 resamples, seed 42; blocks because
the battery carries charge overnight and price regimes last for days). Saved in
`battery_summary_forecast_deg40.csv`.

| Strategy | Capture vs perfect | Net revenue minus naive | Resamples not ahead | Days ahead of naive | Median daily gap to naive |
|---|---|---|---|---|---|
| Forecast, median price | 47.7 % (40.7–53.9) | +$0.55 m (−$2.9 m to +$2.8 m) | 32 % | 64.8 % (60.8–68.8) | +$2,134 ($1,630–2,897) |
| Forecast, linear expected price (a) | 47.7 % (41.6–54.8) | +$0.55 m (−$3.0 m to +$3.0 m) | 33 % | 63.0 % (58.7–67.1) | +$2,045 ($1,328–2,909) |
| Forecast, learned expected price (c) | 47.0 % (41.1–54.0) | +$0.35 m (−$3.1 m to +$2.8 m) | 37 % | 61.1 % (56.8–65.4) | +$2,198 ($1,125–2,926) |
| Naive time-of-day | 46.0 % (40.5–51.1) | – | – | – | – |

Reading the three comparisons with naive:
- **Total revenue: not distinguishable.** Every interval spans zero by a wide margin, and about a
  third of resampled periods have the forecast battery behind. 821 days do not show that it
  earns more than the naive schedule.
- **Typical day: distinguishable.** The share of days ahead is above 50 % and the median daily
  gap is above $0 for all three planning prices; neither interval comes near "no difference".
- The two disagree because the total is decided by a handful of extreme days while the other two
  measures are not. Both are true; which matters depends on whether the question is "is the
  typical day better" or "does it earn more". For a battery it is the second.

Block lengths of 1 and 14 days give almost the same intervals. The intervals only reflect days
like those observed; they say nothing about kinds of day that did not occur. The three planning
prices cannot be ranked against each other on this evidence.

**Findings.**
- **The forecasts beat naive in 8 of 9 quarters, often by 10–17 percentage points, and halve the
  loss-making days (9 % against 18 %).** Pooled, the advantage is only about 1 point because one
  quarter dominates.
- **Two days offset the rest of the period.** In 2025 Q2, 26 June and 12 June (prices up to about
  $15,000/MWh) account for more than the whole quarter's shortfall against naive: about $2.2 m,
  against $2.5 m gained over the other 819 days. This is a description of where the total comes
  from, not a case for setting those days aside: they are exactly the days a battery exists for.
  On 26 June, prices were already $180–400 at midday and reached $3,900–14,000 from 17:00 to 20:30.
  Naive charged at midday regardless of price (its fixed routine) and discharged into $4,000–9,500
  at 17:30–19:30, earning $1.38 m, partly by luck. The forecast battery made small in-and-out trades
  around $300 and was empty when the spike arrived ($0.19 m).
- **Root cause: spike magnitude is badly under-forecast, and the information that was available
  is not used.** At 13:00 the 4-hour-ahead forecast for 17:00 had median $304 and 99th percentile
  $2,008 (actual $9,656), so the expected price looked like an ordinary dear afternoon. The spike
  classifier did flag the risk (P(spike) = 0.43), but the planning price does not use the spike
  probability, and the water-value curve is a fixed average that does not rise on high-risk days.
- **The planning price that was best on RMSE (c) is not the best for the battery.** The median
  and the linear method earn slightly more than (c). With differences this small this may be
  noise, but it illustrates that forecast accuracy and decision value are different criteria
  (decision-focused evaluation).

### 9.4 Ladder steps 1 and 3: threshold execution and a spike-recalibrated price

Following an external review, two changes were tested that need no retraining (development
folds 1–9, same 821 days, $40/MWh wear; `nemspike battery-forecast --price-methods
learned,recalibrated --execution plan,threshold --threshold-only --tag ladder`).

**Step 1, threshold execution.** Re-plan every 5 minutes; from each plan take the marginal value
λ of energy stored after the next interval (LP dual), then act on the *realised* price p of that
interval: discharge at full power if p > λ/η + wear, charge if p < λη, otherwise idle. Under the
price-taker assumption this mirrors a price-contingent offer submitted before the interval, so
the executed interval needs no forecast. When the quarter's end comes within 4 hours, one plan to
the end is made and followed. Benchmark added: **threshold-only**, the same rule with λ taken from
the historical water-value curve (no forecasts), to separate the execution mechanism from the
forecasts.

**Step 3, recalibrated price.** From earlier folds, per horizon: method c's expected price mapped
through an isotonic (monotone) regression of realised price, separately by spike-probability
bucket (0–5, 5–15, 15–35, >35 %; sparse buckets merged).

**Accuracy of the recalibrated price first** (`scripts/analysis/expected_price_methods.py`): much
*worse* than method c. RMSE 443 against 273 at 30 minutes and 376 against 287 at 4 hours;
calibration slope 0.2–0.5 (ideal 1). In the top decile it predicts about $650–700 at 30 minutes
where about $305 materialises. The high-probability buckets are learned from a handful of extreme
episodes, and isotonic regression, unsmoothed, reproduces their top end and projects spike-level
prices onto every later high-risk forecast.

**Battery results** (95 % block-bootstrap intervals, 7-day blocks; differences against naive):

| Strategy | Capture (interval) | Best-10-days capture | Loss-making days | Cycles/day | Total vs naive, $m (interval) | Days ahead of naive (interval) | Median daily gap, $k (interval) |
|---|---|---|---|---|---|---|---|
| Forecast, plan-following, learned (previous headline) | 47.1 % (41.1–54.0) | 35.5 % | 9.4 % | 1.14 | +0.35 (−3.15 to +2.82) | 61 % (57–65) | +2.2 (+1.1 to +2.9) |
| Forecast, plan-following, recalibrated | 40.1 % (33.2–48.6) | 26.9 % | 14.3 % | 1.28 | −1.94 (−6.29 to +1.11) | 58 % (54–63) | +1.2 (+0.5 to +2.2) |
| Threshold, learned (step 1) | 46.2 % (42.3–50.6) | 45.9 % | 20.7 % | 1.98 | +0.06 (−1.85 to +1.66) | 51 % (46–55) | +0.1 (−0.7 to +0.8) |
| Threshold, recalibrated (steps 1 + 3) | 49.1 % (43.2–54.8) | 58.4 % | 20.8 % | 2.00 | +1.01 (−0.34 to +2.45) | 53 % (49–57) | +0.5 (−0.2 to +1.5) |
| Threshold-only (no forecasts) | 17.2 % (12.1–24.7) | 0.2 % | 13.4 % | 0.62 | −9.52 (−16.34 to −4.18) | 35 % (31–39) | −4.0 (−4.9 to −3.1) |
| Naive daily schedule | 46.0 % (40.5–51.2) | 54.0 % | 17.7 % | 0.96 | – | – | – |

(`reports/metrics/dev/battery_summary_ladder_deg40.csv`; per quarter in
`battery_by_fold_ladder_deg40.csv`. The plan-following learned row reproduces section 9.3 exactly.)

Findings:
- **No variant is distinguishable from naive on total revenue**: every total-difference interval
  spans zero except threshold-only, which is clearly worse.
- **Threshold execution captures the extreme days much better** (best-10-days capture 35 % → 46 %
  with the learned price, 58 % with the recalibrated price, against naive's 54 %), and recovers
  2025 Q2 (36 % → 48–61 %, the June spikes). It reacts to the realised spike instead of a
  forecast that under-estimated it.
- **But it over-trades on ordinary days**: loss-making days double (9 % → 21 %), cycling rises from
  1.1 to 2.0 a day, and the typical-day advantage over naive disappears (days ahead 61 % → 51–53 %,
  median gap no longer distinguishable from zero). Likely causes: full-power bang-bang actions
  around a single marginal value, so small noisy price excursions trigger full-size trades; and λ
  taken at the current state of charge only, with no quantity bands.
- **The mechanism is not the source of value: the forecasts are.** Threshold-only, the same rule
  with a historical λ, captures 17 % and loses to naive on every measure, so the forecast-driven
  threshold strategy's 46–49 % comes from the forecasts.
- **Recalibration hurts plan-following and helps threshold execution, unstably.** With threshold
  execution it gives the highest total (49 %) and best-10-days capture (58 %), but its first fold
  (2024 Q2, learned from one quarter of history) captures only 12 %, and its accuracy is far worse
  than method c. The gain looks like an over-cautious price that happens to hold charge for
  spikes, not a better forecast: exactly the kind of result that can fit June 2025.
- **Conclusion so far:** the plan-following strategy with the learned price remains the only one
  that is reliably better than naive on a typical day; threshold execution trades typical-day
  reliability for extreme-day capture. Any refinement (price bands, a deadband, smoothing the
  recalibration) should be chosen on development folds with these same pre-declared measures, and
  will need the holdout to confirm.

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
- Simplified battery: constant efficiency, linear wear cost only (real degradation depends on depth
  of cycle, temperature and age), no auxiliary load, no ramp limits.
- Battery runs continuously within each test quarter but starts and ends each quarter at 50 %; the
  wear cost ($40/MWh) is an illustrative assumption, not a calibrated figure.
- No network constraints, marginal loss factors or intra-regional congestion.
- Risk-neutral planning: the forecast-driven battery plans on the **expected** price because
  revenue is linear in price, so maximising expected revenue means planning on the expected price
  (the median would undervalue the spike tail by $9–12/MWh on our forecasts). It does not
  penalise downside risk, and a single expected price per interval ignores the value of keeping
  flexibility for uncertain spikes (the water value only partly captures this).
- Weather from a single Melbourne point, not a demand- or generation-weighted average.
- Weather is ERA5 reanalysis (published about 5 days late, 25 km grid) standing in for real-time
  observations; no weather *forecasts* are used yet, which limits the 4-hour horizon.
- No AEMO pre-dispatch forecasts as a benchmark or feature yet (a planned extension).
- One region (VIC1) and no interconnector or neighbouring-region information.
- LightGBM hyperparameters are fixed, not tuned; the spike classifier sometimes uses close to the
  2,000-tree cap.
- 4-hour prediction intervals are too narrow (73 % coverage for a nominal 80 %), most in volatile
  quarters.
- Perfect foresight optimises each day in isolation with knowledge of all 288 prices, an upper
  bound no real strategy can reach.
