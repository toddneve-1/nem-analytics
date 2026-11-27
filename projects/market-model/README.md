# NEM market and reliability model

A simplified five-region model of the National Electricity Market (NEM) in
[PyPSA](https://pypsa.org/), solved with the open-source [HiGHS](https://highs.dev/) solver. It
asks an ESOO-style question for **FY2028-29**: with the existing, committed and anticipated
fleet, is expected unserved energy within the reliability standard? It then asks how the answer
changes if coal retires earlier and if southern gas is tight in winter.

Inputs come from AEMO's ISP *Inputs, Assumptions and Scenarios Report* (IASR) workbook and ISP
demand, wind and solar traces. They are parsed into versioned tables committed in
[`inputs/`](inputs/).

**Status: v0.2.** Every element of the assessment is present, each in simplified form; the
simplifications are listed below. v0.2 adds a whole-year water plan for Tasmania and Snowy,
minimum running levels and ramping for coal and gas, and the groundwork for a gas balance built
from GSOO data. See [docs/design_notes.md](docs/design_notes.md) for the reasons behind each
choice, and [docs/data_needed.md](docs/data_needed.md) for the AEMO data still to be added.

## Results (v0.2)

![Expected unserved energy by region and scenario](docs/figures/eue_by_region.png)

**Expected unserved energy, FY2028-29** (% of regional annual energy; standard 0.002 %). In
brackets: outage draws, out of 10, with any unserved energy.

| Region | Workbook coal, gas unconstrained | Workbook coal, tight winter gas | Early coal exit, gas unconstrained | Early coal exit, tight winter gas |
|---|---|---|---|---|
| QLD1 | 0 (0) | 0 (0) | 0.0005 (4) | **0.015** (10) |
| NSW1 | 0 (0) | **0.021** (6) | **0.022** (10) | **2.25** (10) |
| VIC1 | 0 (0) | **0.050** (10) | 0 (0) | **0.082** (10) |
| SA1 | 0 (0) | **0.87** (10) | **0.079** (10) | **7.06** (10) |
| TAS1 | 0 (0) | 0 (0) | 0 (0) | 0 (0) |

Bold: above the reliability standard. Full results: [`results/eue.csv`](results/eue.csv) and,
for every run, [`results/runs.csv`](results/runs.csv).

**Read these as optimistic.** The Tasmanian and Snowy water plan is made with full knowledge of
the year's weather and breakdowns, so the dams save water for exactly the weeks that need it.
Planning on expected conditions instead is the next change (see Assumptions).

### Findings

1. **Base case: within the standard.** No region has unserved energy in any outage draw.
2. **Tight winter gas creates shortfalls in South Australia and Victoria in every draw.** With
   southern gas-powered generation capped at 195 TJ/day from June to August, SA reaches 0.87 %
   and VIC 0.05 %, and NSW exceeds the standard in six draws of ten. The gas limit drives the
   result, not unlucky outages.
3. **Early coal exit alone pushes NSW and SA above the standard.** Retiring Bayswater and Vales
   Point B leaves NSW with only Mt Piper's 1.4 GW of coal. Gas fills most of the gap: southern
   gas burn rises from 9 PJ to 168 PJ a year, and the peak winter day from about 700 to
   1,650 TJ, above the top of the GSOO's range for southern gas-powered generation.
4. **The two risks compound.** With both, unserved energy is far above the sum of the separate
   effects: NSW 2.25 % against 0.021 % + 0.022 %, and SA 7.1 % against 0.87 % + 0.08 %. Early
   coal retirement is workable here only if winter gas is available, and winter gas is exactly
   what the GSOO flags as at risk. That interaction is what an integrated gas–electricity
   assessment exists to catch.

**What changed from v0.1.** Unserved energy is lower in every scenario: the whole-year water
plan lets Tasmania and Snowy save water for winter, where v0.1 forced every store back to its
starting level each week. The base case moves from marginally above the standard (one draw of
ten) to within it. The compounding of coal and gas risk holds in both versions. On/off
decisions for coal and gas barely change unserved energy: in a shortfall week everything runs
flat out, so minimum levels do not bind (checked against exact per-unit decisions on a tight
week).

![Unserved energy in each outage sample](docs/figures/unserved_by_sample.png)

10 % and 50 % POE demand give almost identical results. The 10 % POE trace differs from the
50 % trace only at peaks (about 480 hours in NSW, by up to 1.1 GW, with the same annual energy),
while the shortfalls here are multi-day energy shortages in winter.

### Sense-check against AEMO publications

| Quantity | This model | Published | Source |
|---|---|---|---|
| Victorian reliability, FY2028-29, committed and anticipated fleet | Within the standard (0 % EUE) | No shortfall against the standard until 2030-31 (2025 ESOO, as summarised by the Victorian Auditor-General's Office) | [VAGO](https://audit.vic.gov.au/report/managing-transition-renewable-energy?section=) |
| Peak winter-day gas for southern gas-powered generation (base case) | About 700 TJ/day averaged over outage draws (390 TJ/day with no outages) | 700 to 1,200 TJ/day | [2025 GSOO webinar](https://www.aemo.com.au/-/media/files/gas/national_planning_and_forecasting/gsoo/2025/2025-gsoo-webinar-presentation.pdf) |
| Timing of southern peak-day gas risk | Tight case modelled in winter 2028 and June 2029 | Peak-day shortfall risk from 2029 (2026 GSOO; 2028 in the 2025 GSOO) | [2026 GSOO](https://www.aemo.com.au/-/media/files/gas/national_planning_and_forecasting/gsoo/2026/2026-gas-statement-of-opportunities.pdf) |

With outages, the base case's winter gas peak sits at the bottom of the GSOO range; without
outages it is well below. Both fit a model whose dams plan with perfect foresight. A
region-by-region comparison with the ESOO's published unserved energy figures needs the ESOO
data workbook, which AEMO's site does not serve to automated downloads.

## Method

| Element | Representation (v0.2) |
|---|---|
| Regions | QLD1, NSW1, VIC1, SA1, TAS1, one bus each |
| Interconnectors | Seven inter-regional flow paths with seasonal forward and reverse limits (notional transfer limits plus uplifts from committed network projects); no losses |
| Demand | Hourly operational demand (sent out, as modelled), Step Change scenario, 10 % and 50 % POE, for each of 13 weather reference years (2011–2023) |
| Fleet | Existing, committed and anticipated units (ESOO definition), 610 units from the final 2026 ISP workbook; closures from the workbook, with Eraring closing on 30 April 2029 (2026 GSOO) |
| Thermal | Aggregated by power station; short-run marginal cost = fuel price × heat rate + variable O&M; seasonal ratings; ramp rates from the workbook |
| Minimum running levels | Coal and combined-cycle gas: workbook minimum stable levels (coal 30–51 % of capacity) and assumed minimum up and down times (coal 12 h / 8 h, CCGT 4 h / 4 h), as a linear "share of units online" per station; checked against exact per-unit on/off decisions |
| Wind and solar | Regional capacity factors, built from each unit's own project trace where its name matches (81–100 % of capacity), otherwise the regional mean |
| Storage | Batteries by region, pumped hydro by station (Wivenhoe, Shoalhaven, Kidston), workbook power and energy |
| Hydro | Tasmania and Snowy as seasonal storage: Tasmanian monthly inflows (GWh) from the workbook for the weather year, Snowy's monthly pattern from the workbook scaled to an assumed 13 % capacity factor; storage 14,400 GWh (Tasmania) and one year's output (Snowy), starting and ending the year half full, never below 20 %. Smaller schemes: flat inflow, weekly storage |
| Dispatch | Two stages: a whole-year solve plans Tasmanian and Snowy storage; weekly windows then dispatch, with batteries and pumped hydro returning to their starting level each week |
| Outages | Monte Carlo, unit by unit: two-state Markov chains for full and partial forced outages, from workbook rates and repair times; 2 draws per weather year and POE level (26 per POE level per scenario), with the same draws in every scenario |
| Reliability | EUE = 0.304 × (10 % POE) + 0.696 × (50 % POE), as % of regional annual energy, against the 0.002 % standard |
| Gas | Daily cap on gas burnt by NSW, VIC, SA and TAS gas-powered generation in June to August (limited-gas cases), from the 2026 GSOO southern balance: production + northern supply + storage withdrawal − demand excluding power generation, with colder days leaving less for power stations ([sources](config/gas_balance.yaml)) |

### Scenarios (2 × 2)

| | Gas unconstrained | Tight winter gas |
|---|---|---|
| **Workbook coal closures** | Base case | Gas cap only |
| **Early coal exit** | Gladstone, Callide B, Bayswater and Vales Point B (5.7 GW) closed before the year | Both |

## Assumptions and limitations

These matter for interpreting the numbers. Most of them make the model **optimistic** about
reliability.

* **Thirteen weather years, two outage draws each.** The ESOO uses more outage draws; with
  26 runs per POE level, small EUE values carry wide sampling error. Weather years are the
  2024 ISP traces (2011–2023).
* **Perfect foresight:** inside each week for dispatch, and over the whole year for the
  Tasmanian and Snowy water plan. Real operators plan water without knowing the year's weather
  or breakdowns. The next step is to plan on expected conditions (average availability, other
  weather years) and treat the plan as a target the dispatch may depart from at a cost.
* **Unit commitment is approximate:** a linear "share of units online" per station, with
  assumed minimum up and down times and no start-up costs. On a tight week it matches exact
  per-unit decisions; on/off detail matters more for cost than for unserved energy.
* **No long-duration outages** (a unit out for months) and no intra-regional network limits.
* **Hydro storage assumptions:** Snowy's annual energy, its storage size, and the 50 %
  starting level and 20 % drought floor for both schemes are assumptions (the workbook gives
  Tasmanian inflows but not storage sizes or operating rules).
* **Ten outage samples** per POE level. EUE is a tail statistic, so the sampling error on small
  values is large. Scenario comparisons are more reliable than absolute levels, because every
  scenario uses the same outage draws.
* **Gas balance from charts and interpolation:** several GSOO figures are read from charts
  (about ±25 TJ/day) and 2028–29 demand is interpolated between the GSOO's 2026 and 2030
  values. Storage running down over a winter, and 1-in-20 peak gas demand, are not modelled.
* **Inputs:** the final 2026 ISP workbook (v7.8, June 2026); the traces are from the 2024 ISP.
  Gladstone also closes in 2029 but no date is published in the sources used, so it runs all
  year.

## Reproduce

From this folder:

```bash
uv sync --package nemmodel --extra dev
```

```bash
uv run nemmodel fetch-traces
```

```bash
uv run nemmodel study
```

```bash
uv run nemmodel figures
```

`nemmodel study` runs 80 one-year dispatches across three processes, which takes about 1.5
hours (each process needs up to about 3 GB of memory). The seed is fixed in [`config/assumptions.yaml`](config/assumptions.yaml), so reruns give
the same results. To rebuild the input tables from a workbook:

```bash
uv run nemmodel build-inputs path/to/workbook.xlsx
```

Tests use synthetic data and the committed inputs; no download is needed:

```bash
uv run pytest
```

## Layout

```
config/assumptions.yaml   modelling assumptions not in the workbook
inputs/<version>/         versioned unit and interconnector tables, with a manifest (source, hash)
src/nemmodel/
  inputs.py               IASR workbook -> unit and interconnector tables
  traces.py               ISP traces -> hourly demand and wind/solar capacity factors
  outages.py              Monte Carlo forced outages
  network.py              PyPSA network, windowed solve, gas constraint
  study.py                2 x 2 scenario study and EUE
  figures.py, cli.py
results/                  study outputs (CSV)
docs/                     design notes and figures
```

## Related work

[ISPyPSA](https://github.com/Open-ISP/ISPyPSA) (UNSW CEEM, the openISP project) is an
open-source PyPSA model of the ISP's capacity expansion. This project uses openISP's
[`isp-workbook-parser`](https://github.com/Open-ISP/isp-workbook-parser) and
[`isp-trace-parser`](https://github.com/Open-ISP/isp-trace-parser), and its data mirror, but
addresses a different question: short-term reliability with a gas coupling, rather than
long-term investment.

## Data and attribution

Inputs are derived from AEMO's IASR workbook and ISP traces, © AEMO, used under
[AEMO's copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions),
via the openISP mirror. Processed input tables are committed; raw traces are downloaded by
`nemmodel fetch-traces` and are not committed.
