"""Power and MDE: seeded simulation cross-checked against the closed-form
normal approximation MDE = (z_{1-alpha/2} + z_{power}) * sd / sqrt(n).
Tolerance 15 percent relative: the two methods differ by design, but an
implementation bug (wrong test, wrong sd, wrong n) lands far outside it.
"""

import numpy as np
import pytest

from stats import power


def _diffs() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.normal(0.0, 0.08, 22)


def test_power_increases_with_effect():
    d = _diffs()
    lo = power.simulated_power(d, delta=0.01, seed=20260611)
    hi = power.simulated_power(d, delta=0.10, seed=20260611)
    assert hi > lo
    assert 0.0 <= lo <= hi <= 1.0


def test_mde_simulation_near_normal_approx():
    d = _diffs()
    sim = power.mde(d, target_power=0.80, seed=20260611)
    approx = power.mde_normal_approx(d, target_power=0.80)
    assert sim == pytest.approx(approx, rel=0.15)


def test_seeded_reproducibility():
    d = _diffs()
    assert power.mde(d, seed=20260611) == power.mde(d, seed=20260611)


def test_zero_variance_rejected():
    # All-identical diffs would otherwise report power=1.0 / MDE=0 (false
    # certainty); the preflight rejects instead (review hardening).
    with pytest.raises(ValueError, match="zero-variance"):
        power.simulated_power(np.full(22, 0.05), delta=0.1, seed=20260611)
    with pytest.raises(ValueError, match="zero-variance"):
        power.mde(np.full(22, 0.05), seed=20260611)


def test_too_few_units_rejected():
    with pytest.raises(ValueError, match="diffs"):
        power.mde_normal_approx(np.array([0.05]))
