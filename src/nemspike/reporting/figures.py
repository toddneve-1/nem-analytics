"""Data-overview figures for the README, generated from the cached data by ``nemspike figures``.

Charts use a single series colour, recessive grid and axes, and plain titles that state what is
shown. Price histograms group the extreme tails into labelled overflow bars rather than dropping
them, so every interval is counted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nemspike.timeutils import to_nem_time

BLUE = "#2a78d6"
INK = "#52514e"
DPI = 150


def _style(ax: plt.Axes, title: str, xlabel: str | None, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def monthly_means(prices: pd.DataFrame) -> pd.DataFrame:
    """Mean RRP and demand per calendar month in NEM time."""
    month = to_nem_time(prices["interval_end"]).dt.tz_localize(None).dt.to_period("M")
    return prices.groupby(month)[["rrp", "demand_mw"]].mean()


def monthly_bar(series: pd.Series, title: str, ylabel: str, path: Path) -> Path:
    labels = series.index.strftime("%b %Y")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(range(len(series)), series.to_numpy(), color=BLUE, width=0.8)
    ax.set_xticks(range(0, len(series), 6), labels[::6])
    ax.axhline(0, color=INK, linewidth=0.8)
    _style(ax, title, None, ylabel)
    return _save(fig, path)


def price_histogram(
    rrp: pd.Series, path: Path, lo: float = -200, hi: float = 400, width: float = 20
) -> Path:
    edges = np.arange(lo, hi + width, width)
    counts, _ = np.histogram(rrp, bins=edges)
    counts[-1] -= int((rrp == hi).sum())  # values equal to hi belong to the overflow bar
    below, above = int((rrp < lo).sum()), int((rrp >= hi).sum())
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(edges[:-1], counts, width=width * 0.9, align="edge", color=BLUE)
    for x, n, label in [
        (lo - 3 * width, below, f"< \u2212${-lo:,.0f}"),
        (hi + 2 * width, above, f"≥ ${hi:,.0f}"),
    ]:
        ax.bar(x, n, width=width * 0.9, align="edge", color=INK)
        ax.annotate(
            f"{label}\n{n:,}",
            (x + width * 0.45, n),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
    ax.axvline(0, color=INK, linewidth=0.8, linestyle="--")
    _style(
        ax,
        f"VIC1 spot price: number of 5-minute intervals per ${width:,.0f} bucket",
        "RRP ($/MWh)",
        "Intervals",
    )
    return _save(fig, path)


def demand_histogram(demand: pd.Series, path: Path, width: float = 100) -> Path:
    edges = np.arange(np.floor(demand.min() / width) * width, demand.max() + width, width)
    counts, _ = np.histogram(demand, bins=edges)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(edges[:-1], counts, width=width * 0.9, align="edge", color=BLUE)
    _style(
        ax,
        f"VIC1 operational demand: number of 5-minute intervals per {width:,.0f} MW bucket",
        "Demand (MW)",
        "Intervals",
    )
    return _save(fig, path)


def data_figures(prices: pd.DataFrame, out_dir: Path) -> list[Path]:
    """All data-overview figures. Returns the written paths."""
    m = monthly_means(prices)
    return [
        monthly_bar(
            m["rrp"],
            "VIC1 average spot price (RRP) by month",
            "$/MWh",
            out_dir / "price_monthly.png",
        ),
        monthly_bar(
            m["demand_mw"],
            "VIC1 average operational demand by month",
            "MW",
            out_dir / "demand_monthly.png",
        ),
        price_histogram(prices["rrp"], out_dir / "price_distribution.png"),
        demand_histogram(prices["demand_mw"], out_dir / "demand_distribution.png"),
    ]
