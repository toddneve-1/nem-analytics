# Design notes

Decisions and the reasons for them, kept current as the model changes.

## Scope of v0.1

A deliberately small model with every element of an ESOO-style reliability assessment present:
versioned IASR inputs, hourly time-sequential dispatch, Monte Carlo forced outages, POE-weighted
expected unserved energy, and a gas–electricity coupling. Each element is simplified; the README
lists the simplifications. Later versions deepen the parts that matter most for the answer
(gas calibration, more weather years, unit commitment).

## Inputs

* **Workbook:** the final 2026 ISP workbook (v7.8, 25 June 2026), downloaded by hand (AEMO's
  site refuses automated downloads) and parsed into `inputs/final-2026-isp-v7.8/`. The draft
  (v7.5, via the openISP mirror) is kept in `inputs/draft-2026-isp-v7.5/` for comparison. Final
  vs draft: Eraring back in the fleet (closure 2029), +5.9 GW of batteries, +1.1 GW of solar,
  lower gas running costs; interconnector limits and hydro inflows unchanged.
* **Closures inside the year:** the workbook gives closure years only, and a unit closing in
  calendar 2029 counts as in service. `closure_dates` switches a station off from a given day:
  Eraring from 30 April 2029 (2026 GSOO). Gladstone also closes in 2029 with no published date,
  so it runs all year.
* **Weather years:** the 2024 ISP traces for 13 reference years (2011-2023), from the openISP
  mirror. Raw files (about 1.7 GB a year) are reduced to hourly regional demand and wind and
  solar capacity factors for FY2028-29 (about 2 MB a year, committed under `inputs/weather/`)
  and deleted. AEMO scales every weather year to the same annual demand forecast, so only the
  shape changes between years. Hydro inflows follow each weather year (the workbook has
  Tasmanian and Snowy inflows for 2011-2024).
* **Fleet:** the ESOO definition (existing, committed and anticipated). Committed units count if
  commissioned before 1 July 2028; a unit is in service if its closure year is 2029 or later.
* **Pumped storage labelled "Hydro":** Wivenhoe and Shoalhaven carry storage energy in the
  workbook, so they are reclassified as pumped hydro.
* **Interconnectors:** notional transfer limits plus uplifts from committed and anticipated
  network projects in service before the study year, inter-regional paths only.
* **Wind and solar:** each unit is matched to its own project trace by normalised name;
  unmatched units take the capacity-weighted mean of matched units of the same type in the
  region. Tasmanian solar (one anticipated farm) has no project trace and uses the mean of the
  Tasmanian zone traces.

## Dispatch

* **Two-stage solve (updated after v0.1).** v0.1 solved the year in weekly windows with every
  store, hydro included, returning to its starting level each week, so dams could not save water
  for winter. Measured on the base case: a single whole-year solve takes about 40 seconds, faster
  than 52 weekly solves (about 73 seconds), because building each problem costs more than solving
  it. But whole-year foresight would let batteries and gas plan around every still week of the
  year. The model now uses two stages:
  1. **Plan:** a whole-year solve sets the level of the two large hydro schemes through the
     year. Each starts and ends the year half full and is never drawn below 20 %.
  2. **Dispatch:** weekly windows. Batteries, pumped hydro and small hydro return to their
     starting level each week (a weekly reset does little harm to batteries that cycle daily);
     the large schemes start and end each week at the planned level.

  The plan is solved in 3-hour steps (time series averaged, snapshot weightings of 3 hours) without
  on/off decisions or ramp limits; checked against an hourly plan on the base case: cost within
  0.05 %, Tasmanian levels within 26 GWh (Snowy's path differs more, but many Snowy schedules cost
  almost the same). This cut a run's peak memory from 4.6 GB to 3.0 GB; problems are passed to
  HiGHS in memory rather than through LP files. One year with outages takes about 3.5 minutes. The plan still has perfect foresight of the year's
  weather and outages, which real operators lack, so it remains optimistic.
* **Hydro inflows from the workbook.** Tasmania: monthly inflow energy (GWh) for the whole
  scheme, for the trace weather year (FY2018: 9,122 GWh). Snowy: the workbook gives
  representative monthly inflows in gigalitres for weather variability only, so the model takes
  their monthly shape and scales it to an assumed 13 % capacity factor (about 4.2 TWh a year).
  Snowy is split between its NSW stations (Tumut 3, Upper Tumut, Guthega) and VIC stations
  (Murray 1 and 2), each with its share of the water; in reality they share reservoirs. Smaller
  irrigation and run-of-river stations keep a flat inflow (15 % capacity factor) and weekly
  storage.
* **Storage sizes.** Tasmania 14,400 GWh, derived from the Tasmanian Economic Regulator's
  November 2025 figure (7,347 GWh = 50.9 % of capacity). Snowy has no published energy capacity
  (its active storage is about 5,300 GL, enough for several years), so it is given one year of
  its output. The 50 % starting level and 20 % floor are assumptions, the floor standing in for
  water held back against drought.
* **Minimum running levels, ramping and start-up delays (added after v0.1).** Coal and
  combined-cycle gas stations have, from the workbook, minimum stable levels (coal: the "typical
  lowest band", 30-51 % of capacity) and ramp-up rates (coal: 44-107 % of capacity per hour), plus
  assumed minimum up and down times (coal 12 h and 8 h, CCGT 4 h and 4 h; the workbook has
  none). The minimum down time stands in for start-up delay. Open-cycle gas and reciprocating
  engines start in minutes and stay fully flexible. Each station has an "online share" between
  0 and 1 each hour (PyPSA's linearised unit commitment): for a multi-unit station it reads as
  the share of its units running. It is a linear programme, so it stays fast.

  *Known weakness of the linear version:* a station can run "40 % online", so its minimum can
  shrink to fit a short dip in demand, where real units would have to switch off and then wait
  out their minimum down time. A test (`test_minimum_stable_level_and_down_time`) shows the
  difference on a small case. An exact version, with one on/off decision per unit, is available
  for validation (`unit_commitment.mode: exact_units`).

  *Ramp-down limits are not applied.* A breakdown takes a unit offline at once. With a
  ramp-down limit, an outage starting in the first hour of a weekly window made the window
  infeasible (the unit could not fall from its previous output to zero fast enough); within a
  window the model sees outages coming and ramps down early, but not across a boundary.

  *Breakdowns override the minimum.* A station's minimum is capped each hour at what it can
  produce (after outages and seasonal ratings). Without this, a unit required to stay on for its
  minimum up time but broken down, or rated at zero (Tamar Valley in parts of the year), makes
  the problem infeasible. Each station starts the year free to stay on or switch off.
* **No network losses or intra-regional limits.**
* **Unserved energy** is the output of a VoLL-priced load-shedding generator in each region.
  Where neighbouring regions are both short, how shortfall is shared between them is arbitrary
  (equal cost); EUE summed across the NEM is unaffected.

## Outages

Two independent two-state Markov chains per unit (full and partial), from the workbook's forced
outage rates, mean times to repair and partial derating factors. Hourly time steps cap the
failure probability at one per hour, which is irrelevant at real outage rates (below 20 %).
The workbook's long-duration outages (for example, a unit out for months) are not modelled.

**Common random numbers:** outage sample *s* is the same in every scenario, so scenario
differences are not sampling noise.

## Reliability metric

EUE = 0.304 × mean unserved energy at 10 % POE demand + 0.696 × mean at 50 % POE, as a
percentage of the region's 50 % POE annual operational demand. Compared with the 0.002 %
standard. The Interim Reliability Measure (0.0006 %) is shown for reference.

## Gas coupling

A daily cap on gas burnt by gas-powered generation in NSW, VIC, SA and TAS (the GSOO southern
system) on every day in June, July and August. Heat rates turn MWh into TJ. v0.1 and v0.2 used
an illustrative cap (half the unconstrained model's busiest winter day, 195 TJ/day in v0.2);
v0.3 derives it from the 2026 GSOO, below. The illustrative mode remains
(`gas.tight_case: illustrative`).

## Gas balance (from the 2026 GSOO)

`config/gas_balance.yaml` and `nemmodel.gas` implement the data-based southern winter gas
balance: production + supply from the north + storage withdrawal - other gas demand = gas
available to power stations each day. Other demand is placed between the GSOO's average and
peak winter day using southern electricity demand as a proxy for how cold the day is, so gas
shortage and electricity stress fall on the same days. Storage withdrawal capacity comes from
the workbook's GSOO facility data (now in `inputs/.../gas_infrastructure.csv`). Production,
northern supply and other demand must come from the 2026 GSOO, which AEMO's site will not serve
to automated downloads; until they are filled in, `load_balance` refuses to run and the study
uses the illustrative cap. Days when other demand alone exceeds supply give power stations
nothing; they are counted, and the shortfall to other gas users is reported, not modelled.
Not yet represented: storage running down over a winter (Iona holds about 24 PJ, less than 90
days at its 570 TJ/day withdrawal rate).

## Results in v0.3 and what they say about the model

With the final workbook fleet, the base case burns only 1.4 PJ of southern gas for power in a
year and the GSOO limit almost never binds. The real NEM burns far more: gas runs for system
strength, network support and contracts, none of which a short-run-cost dispatch represents.
So the gas limit is understated in the workbook-coal cases; the early-exit cases, where gas is
needed for energy, are the informative ones. A minimum gas running requirement by region (for
example from AEMO's system strength requirements) would be the natural next refinement.

## Parallel runs

The study uses a process pool started with "spawn". The parent process has already run HiGHS
for the calibration solve, and forked children deadlocked. Each finished run is appended to
`results/runs_partial.jsonl` with a fingerprint of the configuration and inputs, so a study
stopped by a time limit (`--max-minutes`) or a crash resumes without repeating runs.
