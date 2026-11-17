from __future__ import annotations

import numpy as np
import pytest

from nemspike.battery.expected_price import LearnedBands, expected_price, linear_expected


def test_linear_expected_of_symmetric_uniform_is_its_mean() -> None:
    # Uniform(0, 100): q0.1 = 10, q0.5 = 50, q0.9 = 90, q0.99 = 99.
    q = np.array([[10.0, 50.0, 90.0, 99.0]])
    # Tails at the outer quantiles: 0.1*10 + 0.4*30 + 0.4*70 + 0.09*94.5 + 0.01*99 = 50.495
    assert linear_expected(q)[0] == pytest.approx(50.495)


def test_learned_recovers_mean_of_skewed_distribution() -> None:
    """On a right-skewed distribution with calibrated quantiles, the learned method should get the
    mean right where the median does not."""
    rng = np.random.default_rng(0)
    y = rng.lognormal(mean=4.0, sigma=0.8, size=200_000)
    true_q = np.quantile(y, [0.1, 0.5, 0.9, 0.99])
    q = np.tile(true_q, (len(y), 1))
    bands = LearnedBands.fit(q, y)
    np.testing.assert_allclose(bands.probs, [0.1, 0.4, 0.4, 0.09, 0.01], atol=0.002)
    est = bands.expected(q[:1])[0]
    assert est == pytest.approx(y.mean(), rel=0.01)
    assert true_q[1] < 0.8 * y.mean()  # the median is well below the mean here


def test_learned_corrects_too_narrow_intervals() -> None:
    """If forecast intervals are too narrow, outcomes fall outside them more often than nominal;
    the learned band probabilities should reflect that."""
    rng = np.random.default_rng(1)
    y = rng.normal(100, 30, 50_000)
    narrow = np.tile([100 - 20, 100, 100 + 20, 100 + 40], (len(y), 1))  # too narrow
    bands = LearnedBands.fit(narrow, y)
    assert bands.probs[0] > 0.2 and bands.probs[4] > 0.05
    assert bands.expected(narrow[:1])[0] == pytest.approx(100, abs=1.0)


def test_expected_price_dispatch_and_errors() -> None:
    q = np.array([[10.0, 50.0, 90.0, 99.0]])
    assert expected_price(q, "median")[0] == 50.0
    with pytest.raises(ValueError):
        expected_price(q, "learned")
    with pytest.raises(ValueError):
        expected_price(q, "nope")  # type: ignore[arg-type]
