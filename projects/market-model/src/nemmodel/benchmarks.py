"""AEMO publications the model is compared with.

``esoo_unserved_energy`` reads the "Unserved Energy" sheet of the 2026 ESOO results workbook:
expected unserved energy by region and financial year for each reliability assessment
(committed and anticipated developments; government schemes and actionable developments; and
sensitivities). Values in the workbook are fractions of annual energy; they are returned as
percentages, like the model's results.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd

ESOO_REGIONS = {"New South Wales": "NSW1", "Queensland": "QLD1", "South Australia": "SA1",
                "Tasmania": "TAS1", "Victoria": "VIC1"}  # fmt: skip


def esoo_unserved_energy(workbook: Path) -> pd.DataFrame:
    """Long table: assessment, financial_year, region, eue_pct (blank cells are skipped)."""
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    out, assessment, columns = [], None, None
    for row in wb["Unserved Energy"].iter_rows(values_only=True):
        cells = {i: v for i, v in enumerate(row) if v not in (None, "")}
        if not cells:
            continue
        first = str(cells[min(cells)])
        if first == "Financial Year":
            columns = {i: str(v) for i, v in cells.items()}  # keep column positions
        elif columns and first[:2] == "20" and "-" in first:
            for i, v in cells.items():
                name = columns.get(i)
                if name in ESOO_REGIONS:
                    out.append({
                        "assessment": assessment, "financial_year": first,
                        "region": ESOO_REGIONS[name], "eue_pct": 100 * float(v),
                    })  # fmt: off
        elif len(cells) == 1 and not first.startswith(("Unserved", "Data shows", "For ")):
            assessment, columns = first, None  # a new block starts with its title
    return pd.DataFrame(out)
