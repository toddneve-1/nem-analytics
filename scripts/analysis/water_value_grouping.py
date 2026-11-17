"""Does the value of stored energy differ by weekday/weekend, or by day of week?

The marginal value of stored energy at an interval is the shadow price (dual) of the
energy-balance constraint in the perfect-foresight LP: what one extra MWh in storage at that
interval would add to revenue ($/MWh). It is computed quarter by quarter (continuous operation,
battery and wear cost from the config) for 2023-2025 with
:func:`nemspike.battery.water_value.storage_values`, which removes the rare negative-price
simultaneous charge/discharge before taking duals.

Average curves by time of day are learned on 2023-2024 with three groupings and scored on how well
they predict the 2025 values (mean absolute error). The test year is never used for learning.

Run:  uv run python scripts/analysis/water_value_grouping.py
Writes: reports/metrics/dev/water_value_grouping.csv and water_value_by_day.csv
"""

from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from nemspike.battery.lp import BatterySpec
from nemspike.battery.water_value import storage_values
from nemspike.config import load_config
from nemspike.data.aemo import load_price_demand
from nemspike.timeutils import INTERVAL, to_nem_time

OUT_DIR = Path("reports/metrics/dev")
LAST_DAY = pd.Timestamp("2025-12-31 14:00", tz="UTC")  # 24:00 AEST on 31 Dec 2025


def main() -> None:
    cfg = load_config()
    spec = BatterySpec.from_config(cfg.battery)
    prices = load_price_demand(cfg.data.start, cfg.data.end, cfg.data.region, cfg.data.raw_dir)
    prices = prices[prices["interval_end"] <= LAST_DAY].copy()
    start = to_nem_time(prices["interval_end"] - INTERVAL).dt.tz_localize(None)
    prices["quarter"] = start.dt.to_period("Q").to_numpy()

    values = [
        pd.Series(
            storage_values(chunk["rrp"].to_numpy(), spec, soc_end=spec.soc_initial_mwh),
            index=chunk.index,
        )
        for _, chunk in prices.groupby("quarter", sort=True)
    ]
    prices["value"] = pd.concat(values)
    vic = holidays.Australia(subdiv="VIC", years=range(2023, 2026))
    prices["slot"] = start.dt.hour * 12 + start.dt.minute // 5
    prices["dow"] = start.dt.day_of_week
    prices["holiday"] = np.array([d in vic for d in start.dt.date])
    prices["weekend"] = ((prices["dow"] >= 5) | prices["holiday"]).astype(int)
    prices["year"] = start.dt.year

    train, test = prices[prices["year"] <= 2024], prices[prices["year"] == 2025]
    rows = []
    for name, keys in [
        ("time of day only", ["slot"]),
        ("time of day x weekday/weekend (holidays as weekend)", ["slot", "weekend"]),
        ("time of day x day of week", ["slot", "dow"]),
    ]:
        for stat in ("mean", "median"):
            table = train.groupby(keys)["value"].agg(stat).rename("pred")
            pred = test.join(table, on=keys)["pred"]
            rows.append(
                {
                    "grouping": name,
                    "statistic": stat,
                    "groups": len(table),
                    "test_mae": float(np.mean(np.abs(test["value"] - pred))),
                    "test_rmse": float(np.sqrt(np.mean((test["value"] - pred) ** 2))),
                }
            )
    result = pd.DataFrame(rows)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_day = (
        prices.assign(hour=prices["slot"] // 12)
        .pivot_table(index="hour", columns="dow", values="value", aggfunc="mean")
        .rename(columns=dict(enumerate(names)))
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_DIR / "water_value_grouping.csv", index=False)
    by_day.to_csv(OUT_DIR / "water_value_by_day.csv")

    print(
        f"Storage value 2023-2025: median ${prices['value'].median():.0f}/MWh, 10th-90th "
        f"percentile ${prices['value'].quantile(0.1):.0f} to ${prices['value'].quantile(0.9):.0f}"
    )
    print(result.round(2).to_string(index=False))
    daily = prices.groupby("dow")["value"].mean().rename(index=dict(enumerate(names)))
    print("Average by day of week ($/MWh):", daily.round(0).to_dict())
    print(by_day.loc[[8, 13, 18, 22]].round(0).to_string())


if __name__ == "__main__":
    main()
