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

* **Linear programme, weekly windows.** A full-year LP takes about 75 seconds, so the study
  (80 runs) takes about 25 minutes on four cores. Weekly windows keep each LP small. Storage
  (including conventional hydro) is cyclic within each window: a battery cannot carry energy
  from one week to the next, and hydro uses one week of average inflow per week. This is
  simpler than ESOO practice (which models hydro storage over the year) and is optimistic
  within a week because of perfect foresight.
* **No unit commitment.** No minimum loads, ramp limits or start costs. Coal can follow demand
  freely, which understates the need for flexible gas.
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

## Parallel runs

The study uses a process pool started with "spawn". The parent process has already run HiGHS
for the calibration solve, and forked children deadlocked.
