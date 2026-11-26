# NEM market and reliability model

A simplified five-region model of the National Electricity Market (NEM) in
[PyPSA](https://pypsa.org/), solved with the open-source [HiGHS](https://highs.dev/) solver. It
asks an ESOO-style question for **FY2028-29**: with the existing, committed and anticipated
fleet, is expected unserved energy within the reliability standard? It then asks how the answer
changes if coal retires earlier and if southern gas is tight in winter.

Inputs come from AEMO's ISP *Inputs, Assumptions and Scenarios Report* (IASR) workbook and ISP
demand, wind and solar traces. They are parsed into versioned tables committed in
[`inputs/`](inputs/).

**Status: v0.1.** Every element of the assessment is present, each in simplified form; the
simplifications are listed below. See [docs/design_notes.md](docs/design_notes.md) for the
reasons behind each choice.

<!-- RESULTS -->

## Method

| Element | v0.1 representation |
|---|---|
| Regions | QLD1, NSW1, VIC1, SA1, TAS1, one bus each |
| Interconnectors | Seven inter-regional flow paths with seasonal forward and reverse limits (notional transfer limits plus uplifts from committed network projects); no losses |
| Demand | Hourly operational demand (sent out, as modelled), Step Change scenario, 10 % and 50 % POE, weather reference year 2018 |
| Fleet | Existing, committed and anticipated units (ESOO definition), 573 units; closures from the workbook |
| Thermal | Aggregated by power station; short-run marginal cost = fuel price × heat rate + variable O&M; seasonal ratings |
| Wind and solar | Regional capacity factors, built from each unit's own project trace where its name matches (81–100 % of capacity), otherwise the regional mean |
| Storage | Batteries by region, pumped hydro by station (Wivenhoe, Shoalhaven, Kidston), workbook power and energy |
| Hydro | By region, as storage with a constant inflow sized to an annual capacity factor (Tasmania 45 %, mainland 12–15 %) |
| Dispatch | Linear programme, solved in weekly windows with perfect foresight; storage returns to its starting level each week |
| Outages | Monte Carlo, unit by unit: two-state Markov chains for full and partial forced outages, from workbook rates and repair times; 10 samples per POE per scenario, with the same draws in every scenario |
| Reliability | EUE = 0.304 × (10 % POE) + 0.696 × (50 % POE), as % of regional annual energy, against the 0.002 % standard |
| Gas | Daily cap on gas burnt by NSW, VIC, SA and TAS gas-powered generation in June to August (tight case only) |

### Scenarios (2 × 2)

| | Gas unconstrained | Tight winter gas |
|---|---|---|
| **Workbook coal closures** | Base case | Gas cap only |
| **Early coal exit** | Gladstone, Callide B, Bayswater and Vales Point B (5.7 GW) closed before the year | Both |

## Assumptions and limitations

These matter for interpreting the numbers. Most of them make the model **optimistic** about
reliability.

* **One weather year.** The ESOO samples many reference years; v0.1 uses 2018 only. Weather
  variability, the largest driver of EUE, is not sampled.
* **Perfect foresight inside each week** and storage that is free to optimise against it.
* **No unit commitment:** no minimum loads, ramp rates or start-up times.
* **No long-duration outages** (a unit out for months) and no intra-regional network limits.
* **Hydro energy is spread evenly** across the year rather than following seasonal inflows and
  storage targets.
* **Ten outage samples** per POE level. EUE is a tail statistic, so the sampling error on small
  values is large. Scenario comparisons are more reliable than absolute levels, because every
  scenario uses the same outage draws.
* **The tight-gas cap is illustrative:** half of the model's own peak winter-day gas burn. It is
  not yet derived from the GSOO supply-demand balance.
* **Draft inputs.** The workbook is the Draft 2026 ISP version (v7.5); the traces are from the
  2024 ISP.

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

`nemmodel study` runs 80 one-year dispatches across four processes, which takes about half an
hour. The seed is fixed in [`config/assumptions.yaml`](config/assumptions.yaml), so reruns give
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
