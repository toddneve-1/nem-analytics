# Design notes

Decisions and the reasons for them, kept current as the model changes.

## Scope of v0.1

A deliberately small model with every element of an ESOO-style reliability assessment present:
versioned IASR inputs, hourly time-sequential dispatch, Monte Carlo forced outages, POE-weighted
expected unserved energy, and a gas–electricity coupling. Each element is simplified; the README
lists the simplifications. Later versions deepen the parts that matter most for the answer
(gas calibration, more weather years, unit commitment).

## Inputs

* **Workbook:** Draft 2026 ISP IASR workbook (v7.5), from the openISP mirror, because AEMO's site
  refuses automated downloads. The final 2026 ISP workbook (v7.8) should replace it when a copy
  can be downloaded by hand; `nemmodel build-inputs` regenerates the tables and the manifest
  records the file hash.
* **Traces:** 2024 ISP traces for reference year 2018 (the openISP "example" set). Only one
  weather year, so weather variability is not sampled; this understates EUE.
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
  lowest band", 30-51 % of capacity) and ramp rates (coal: 44-107 % of capacity per hour), plus
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
system) on every day in June, July and August. Heat rates turn MWh into TJ. The tight-winter cap
is half the most the unconstrained model burns on a winter day (workbook fleet, 10 % POE, no
outages): 587 TJ/day against 1,174 TJ/day. The unconstrained peak sits at the top of the 2025
GSOO's 700–1,200 TJ/day range for southern GPG demand, so the reference is plausible. The cap
itself is an illustrative stress case; v0.2 should derive it from southern supply capacity less
non-GPG peak-day demand, using the GSOO facility data in the workbook.

## Gas balance (groundwork, waiting on GSOO data)

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

## Parallel runs

The study uses a process pool started with "spawn". The parent process has already run HiGHS
for the calibration solve, and forked children deadlocked.
