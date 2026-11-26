"""Compare ways of turning quantile forecasts (q0.1, q0.5, q0.9, q0.99) and the spike probability
into an expected price, against realised prices.

The expected price is the probability-weighted average of the price distribution, equivalently
the average of the quantile function Q(u) over u in [0, 1]. With four quantiles, two things are
unknown: where the mass sits inside each band, and the tails beyond q0.1 and q0.99.

Methods (all leakage-free):
  a  linear: each band's mean is its midpoint; tails taken at q0.1 and q0.99
  b  asinh: Q(u) linear in asinh(price / 100) between quantiles, tails extrapolated with the
     nearest segment's slope; band means in closed form
  c  learned: from EARLIER folds' out-of-sample results, the empirical probability of each band,
     the mean position of outcomes inside each interior band, and the mean excess beyond q0.1 and
     q0.99 (also corrects for mis-calibrated intervals)
  d  hurdle: (1 - p_spike) * mean of the quantiles clipped at the spike threshold
     + p_spike * mean spike size from EARLIER folds
  e  recalibrated: method c's expected price mapped, per spike-probability bucket, through an
     isotonic regression of realised price learned from EARLIER folds (uses the spike classifier)
  median: q0.5, for reference

Scored on development folds 1-9 (fold 0 has no earlier history for c and d), all horizons.
RMSE is the headline: the mean is the forecast that minimises squared error.

Run:  uv run python scripts/analysis/expected_price_methods.py
Writes: reports/metrics/dev/expected_price_methods.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nemspike.battery.expected_price import LearnedBands, SpikeRecalibration

PRED_DIR = Path("artefacts/backtest/dev")
OUT = Path("reports/metrics/dev/expected_price_methods.csv")
QS = ["q0.1", "q0.5", "q0.9", "q0.99"]
U = np.array([0.1, 0.5, 0.9, 0.99])
WIDTHS = np.array([0.1, 0.4, 0.4, 0.09, 0.01])  # below q0.1, three interior bands, above q0.99
SCALE = 100.0


def method_a(q: np.ndarray) -> np.ndarray:
    q1, q5, q9, q99 = q.T
    return 0.1 * q1 + 0.4 * (q1 + q5) / 2 + 0.4 * (q5 + q9) / 2 + 0.09 * (q9 + q99) / 2 + 0.01 * q99


def _mean_sinh(za: np.ndarray, zb: np.ndarray) -> np.ndarray:
    """Average of SCALE * sinh(z) for z linear from za to zb (closed form)."""
    dz = zb - za
    safe = np.abs(dz) > 1e-9
    out = SCALE * np.sinh(za)
    out[safe] = SCALE * (np.cosh(zb[safe]) - np.cosh(za[safe])) / dz[safe]
    return out


def method_b(q: np.ndarray) -> np.ndarray:
    z = np.arcsinh(q / SCALE)
    slope_lo = (z[:, 1] - z[:, 0]) / (U[1] - U[0])
    slope_hi = (z[:, 3] - z[:, 2]) / (U[3] - U[2])
    knots = [
        z[:, 0] - slope_lo * U[0],  # extrapolated to u = 0
        z[:, 0],
        z[:, 1],
        z[:, 2],
        z[:, 3],
        z[:, 3] + slope_hi * (1 - U[3]),  # extrapolated to u = 1
    ]
    return sum(WIDTHS[i] * _mean_sinh(knots[i], knots[i + 1]) for i in range(5))


def fit_c(hist: pd.DataFrame) -> dict[str, np.ndarray | float]:
    """Band probabilities, interior positions and tail excesses from earlier out-of-sample rows."""
    y, q = hist["y"].to_numpy(), hist[QS].to_numpy()
    band = np.select([y < q[:, 0], y < q[:, 1], y < q[:, 2], y <= q[:, 3]], [0, 1, 2, 3], default=4)
    probs = np.bincount(band, minlength=5) / len(y)
    pos = np.zeros(3)
    for j in range(3):
        lo, hi = q[:, j], q[:, j + 1]
        m = (band == j + 1) & (hi - lo > 1e-6)
        pos[j] = np.mean((y[m] - lo[m]) / (hi[m] - lo[m])) if m.any() else 0.5
    low_excess = float(np.mean(y[band == 0] - q[band == 0, 0])) if (band == 0).any() else 0.0
    high_excess = float(np.mean(y[band == 4] - q[band == 4, 3])) if (band == 4).any() else 0.0
    return {"probs": probs, "pos": pos, "low": low_excess, "high": high_excess}


def method_c(q: np.ndarray, p: dict) -> np.ndarray:
    probs, pos = p["probs"], p["pos"]
    means = [q[:, 0] + p["low"]]
    means += [q[:, j] + pos[j] * (q[:, j + 1] - q[:, j]) for j in range(3)]
    means += [q[:, 3] + p["high"]]
    return sum(probs[i] * means[i] for i in range(5))


def method_d(q: np.ndarray, spike_prob: np.ndarray, threshold: np.ndarray, spike_mean: float):
    clipped = np.minimum(q, threshold[:, None])
    return (1 - spike_prob) * method_a(clipped) + spike_prob * spike_mean


def score(y: np.ndarray, e: np.ndarray) -> dict[str, float]:
    err = e - y
    deciles = pd.qcut(e, 10, labels=False, duplicates="drop")
    top = deciles == deciles.max()
    slope = float(np.polyfit(e, y, 1)[0])
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "calibration_slope": slope,
        "top_decile_pred": float(np.mean(e[top])),
        "top_decile_actual": float(np.mean(y[top])),
    }


def main() -> None:
    rows = []
    for path in sorted(
        PRED_DIR.glob("h*/lgbm_asinh.parquet"), key=lambda p: int(p.parent.name[1:])
    ):
        h = int(path.parent.name[1:])
        df = pd.read_parquet(path).sort_values("origin")
        parts = []
        for fold in sorted(df["fold"].unique())[1:]:
            hist, cur = df[df["fold"] < fold], df[df["fold"] == fold]
            q = cur[QS].to_numpy()
            thr = cur["spike_threshold"].to_numpy()
            spike_mean = float(hist.loc[hist["y"] > hist["spike_threshold"], "y"].mean())
            bands = LearnedBands.fit(hist[QS].to_numpy(), hist["y"].to_numpy())
            recal = SpikeRecalibration().fit(
                bands.expected(hist[QS].to_numpy()),
                hist["spike_prob"].to_numpy(),
                hist["y"].to_numpy(),
            )
            recalibrated = recal.predict(bands.expected(q), cur["spike_prob"].to_numpy())
            parts.append(
                pd.DataFrame(
                    {
                        "y": cur["y"].to_numpy(),
                        "median": q[:, 1],
                        "a_linear": method_a(q),
                        "b_asinh": method_b(q),
                        "c_learned": method_c(q, fit_c(hist)),
                        "d_hurdle": method_d(q, cur["spike_prob"].to_numpy(), thr, spike_mean),
                        "e_recalibrated": recalibrated,
                    }
                )
            )
        res = pd.concat(parts, ignore_index=True)
        y = res["y"].to_numpy()
        for m in ["median", "a_linear", "b_asinh", "c_learned", "d_hurdle", "e_recalibrated"]:
            rows.append({"horizon": h, "method": m, "n": len(y)} | score(y, res[m].to_numpy()))
        rows.append(
            {
                "horizon": h,
                "method": "actual_mean",
                "n": len(y),
                "bias": 0.0,
                "top_decile_actual": float(np.mean(y)),
            }
        )
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(out[out.method != "actual_mean"].round(3).to_string(index=False))
    means = out[out.method == "actual_mean"].set_index("horizon")["top_decile_actual"]
    print("\nactual mean price by horizon rows:", means.round(2).to_dict())


if __name__ == "__main__":
    main()
