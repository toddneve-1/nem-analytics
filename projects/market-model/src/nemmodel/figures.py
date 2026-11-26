"""README figures, drawn from saved results (``results/eue.csv`` and ``results/runs.csv``)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nemmodel.network import REGIONS

CASES = [
    "coal-workbook_gas-unconstrained",
    "coal-workbook_gas-tight",
    "coal-early_gas-unconstrained",
    "coal-early_gas-tight",
]
LABELS = {
    "coal-workbook_gas-unconstrained": "Workbook coal, gas unconstrained",
    "coal-workbook_gas-tight": "Workbook coal, tight winter gas",
    "coal-early_gas-unconstrained": "Early coal exit, gas unconstrained",
    "coal-early_gas-tight": "Early coal exit, tight winter gas",
}
COLOURS = dict(zip(CASES, ["#257de0", "#f47230", "#00b894", "#f2a600"], strict=True))
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def eue_chart(eue: pd.DataFrame, cfg: dict, path: Path) -> None:
    """EUE (% of regional energy) by region for each case, against the reliability standard."""
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
    width = 0.2
    x = np.arange(len(REGIONS))
    t = eue.set_index(["case", "region"])["eue_pct"]
    floor = 1e-4  # log axis: zero EUE is drawn as a "0" label at the floor
    for i, case in enumerate(CASES):
        vals = np.array([t.get((case, r), np.nan) for r in REGIONS])
        xs = x + (i - 1.5) * width
        ax.bar(
            xs,
            np.where(vals > floor, vals, np.nan),
            width * 0.9,
            bottom=floor,
            color=COLOURS[case],
            label=LABELS[case],
            edgecolor="white",
            linewidth=1,
        )
        for xi, v in zip(xs, vals, strict=True):
            if not v > floor:
                ax.text(xi, floor * 1.15, "0", ha="center", va="bottom", fontsize=7, color=MUTED)
    std = cfg["reliability"]["standard_pct"]
    ax.axhline(std, color=TEXT, linewidth=1, linestyle="--")
    ax.annotate(
        f"Reliability standard {std:g} %",
        (len(REGIONS) - 0.5, std),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        color=TEXT,
        fontsize=8,
    )
    ax.set_yscale("log")
    ax.set_ylim(floor, 30)
    ax.set_xlim(-0.5, len(REGIONS) - 0.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(x, REGIONS)
    ax.set_ylabel("Expected unserved energy (% of annual energy, log scale)", color=TEXT)
    ax.set_title("Expected unserved energy, FY2028-29", color=TEXT, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def samples_chart(runs: pd.DataFrame, path: Path) -> None:
    """NEM-wide unserved energy in each outage sample, by case and demand POE."""
    runs = runs.assign(total=runs[[f"unserved_mwh_{r}" for r in REGIONS]].sum(axis=1) / 1000)
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
    rng = np.random.default_rng(0)
    for i, case in enumerate(CASES):
        for poe, marker, dx in (("POE50", "o", -0.15), ("POE10", "^", 0.15)):
            v = runs.loc[(runs["case"] == case) & (runs["poe"] == poe), "total"]
            jitter = rng.uniform(-0.06, 0.06, len(v))
            ax.scatter(i + dx + jitter, v, s=28, marker=marker, color=COLOURS[case],
                       edgecolor="white", linewidth=1,
                       label=f"{poe[:3]} {poe[3:]} %" if i == 0 else None)  # fmt: off
    ax.set_xticks(range(len(CASES)), [LABELS[c].replace(", ", ",\n") for c in CASES], fontsize=8)
    ax.set_ylabel("Unserved energy, NEM (GWh)", color=TEXT)
    ax.set_title("Unserved energy in each outage sample (circles 50 % POE, triangles 10 % POE)",
                 color=TEXT, loc="left", fontsize=11)  # fmt: off
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_all(root: Path, results: Path) -> None:
    import yaml

    cfg = yaml.safe_load((root / "config" / "assumptions.yaml").read_text())
    out = root / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    eue_chart(pd.read_csv(results / "eue.csv"), cfg, out / "eue_by_region.png")
    samples_chart(pd.read_csv(results / "runs.csv"), out / "unserved_by_sample.png")
