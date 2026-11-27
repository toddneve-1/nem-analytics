# Data still needed

What the model would use but does not yet have, who can get it, and where it goes. AEMO's
website refuses automated downloads, so AEMO files must be downloaded in a browser and added
by hand. Files on the openISP mirror (data.openisp.au) can be fetched by the code.

| Item | Why | Who | Status |
|---|---|---|---|
| **2026 GSOO report and data file** (AEMO, March 2026) | Southern gas supply for 2028-29: production capacity by field, and gas demand other than power generation (homes, commerce, industry) on winter days. Replaces the illustrative gas cap with a gas balance built from data | Download by hand (AEMO blocks automated access) and add to `data/raw/gsoo/` | Needed |
| **Final 2026 ISP IASR workbook** (v7.8) | Replace the draft workbook (v7.5). Rebuild inputs with `nemmodel build-inputs` | Download by hand from AEMO's ISP page | Needed |
| **2025 (or latest) ESOO data workbook** | Region-by-region comparison of expected unserved energy with AEMO's published figures | Download by hand from AEMO's ESOO page | Needed |
| **More weather years** (2024 ISP traces, reference years 2011-2023) | One weather year understates risk; the ESOO uses many. About 1-1.7 GB raw per year, so each year will be processed to hourly regional series and the raw files deleted | The code can fetch these from openISP | Available; not yet added (run time grows with each year) |

Notes

* The gas balance code checks for the GSOO values and stops with a clear message if they are
  missing, rather than running on placeholders.
* Hourly series extracted from the traces are small enough to commit under `inputs/`, so the
  study can rerun without downloading the raw traces again.
