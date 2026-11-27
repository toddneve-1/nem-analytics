# Data still needed

What the model would use but does not yet have, who can get it, and where it goes. AEMO's
website refuses automated downloads, so AEMO files must be downloaded in a browser and added
by hand. Files on the openISP mirror (data.openisp.au) can be fetched by the code.

| Item | Why | Who | Status |
|---|---|---|---|
| **2026 GSOO report** (AEMO, March 2026) — **received and used** (`config/gas_balance.yaml`); data file optional, would replace chart readings | Southern gas supply for 2028-29: production capacity by field, and gas demand other than power generation (homes, commerce, industry) on winter days. Replaces the illustrative gas cap with a gas balance built from data | Download by hand (AEMO blocks automated access) and add to `data/raw/gsoo/` | Needed |
| **Final 2026 ISP IASR workbook** (v7.8) | Replace the draft workbook (v7.5). Rebuild inputs with `nemmodel build-inputs` | Download by hand from AEMO's ISP page | **Received** (25 June 2026); inputs in `inputs/final-2026-isp-v7.8/` |
| **2026 ESOO results workbook** | Region-by-region comparison of expected unserved energy with AEMO's published figures | Download by hand from AEMO's ESOO page | **Received** (August 2026) |
| **More weather years** (2024 ISP traces, reference years 2011-2023) | One weather year understates risk; the ESOO uses many. About 1-1.7 GB raw per year, so each year will be processed to hourly regional series and the raw files deleted | The code can fetch these from openISP | Available; not yet added (run time grows with each year) |

## The GSOO values the gas balance needs

Fill these into [`config/gas_balance.yaml`](../config/gas_balance.yaml), each for winter 2028
(July-August) and winter 2029 (June), then set `gas.tight_case: gsoo` in
`config/assumptions.yaml`:

| Field | What to look for in the 2026 GSOO |
|---|---|
| `southern_production_tj_d` | Peak-day production capacity of southern fields (Gippsland, Otway, Bass, Cooper basin output available to the south) |
| `northern_supply_tj_d` | Gas from Queensland available to the southern states on a peak day, after LNG export commitments (via the Moomba to Sydney and South West Queensland pipelines) |
| `other_demand_peak_tj_d` | Southern peak-day demand excluding gas-powered generation (residential, commercial, industrial), 1-in-2 peak |
| `other_demand_average_tj_d` | Southern average winter-day demand excluding gas-powered generation |
| `source` | Where each figure came from (figure or sheet) |

Storage withdrawal (Iona, Dandenong LNG, Newcastle, Moomba) comes from the workbook
automatically (`inputs/.../gas_infrastructure.csv`).

## Notes

* The gas balance code checks for the GSOO values and stops with a clear message if they are
  missing, rather than running on placeholders.
* Hourly series extracted from the traces are small enough to commit under `inputs/`, so the
  study can rerun without downloading the raw traces again.
