"""Simulation-based power and MDE from measured per-question differences.

The simulation resamples the observed differences (preserving their measured
variance), shifts by delta, and tests at alpha with a paired t-test; the
closed-form normal approximation is the cross-check, per the v3.1 plan WP3.
The simulation is vectorized: one (n_sim, n) resample tested with axis=1.

Pure module: stdlib + numpy + scipy only (guardrail 1).
"""

import numpy as np
from scipy.stats import norm, ttest_1samp

from stats._validate import require_finite, require_min_units

DEFAULT_SEED = 20260611
DEFAULT_N_SIM = 2_000


def simulated_power(
    diffs: np.ndarray,
    delta: float,
    alpha: float = 0.05,
    n_sim: int = DEFAULT_N_SIM,
    seed: int = DEFAULT_SEED,
) -> float:
    centered = require_finite(diffs, "diffs")
    require_min_units(len(centered), 2, "diffs")
    centered = centered - centered.mean()
    # < tol, not == 0: np.std of identical values leaves ~1e-17 float noise.
    if centered.std(ddof=1) < 1e-12:
        raise ValueError("zero-variance diffs; power is not estimable")
    rng = np.random.default_rng(seed)
    n = len(centered)
    samples = rng.choice(centered, size=(n_sim, n), replace=True) + delta
    pvals = ttest_1samp(samples, 0.0, axis=1).pvalue
    return float(np.mean(pvals < alpha))


def mde(
    diffs: np.ndarray,
    target_power: float = 0.80,
    alpha: float = 0.05,
    n_sim: int = DEFAULT_N_SIM,
    seed: int = DEFAULT_SEED,
    tol: float = 1e-3,
) -> float:
    d = require_finite(diffs, "diffs")
    require_min_units(len(d), 2, "diffs")
    sd = float(np.std(d, ddof=1))
    if sd < 1e-12:  # not == 0: np.std of identical values leaves float noise
        raise ValueError("zero-variance diffs; MDE is not estimable")
    lo, hi = 0.0, sd * 4 + 1e-6
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if simulated_power(d, mid, alpha=alpha, n_sim=n_sim, seed=seed) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi


def mde_normal_approx(diffs: np.ndarray, target_power: float = 0.80, alpha: float = 0.05) -> float:
    d = require_finite(diffs, "diffs")
    require_min_units(len(d), 2, "diffs")
    sd = float(d.std(ddof=1))
    if sd < 1e-12:  # not == 0: np.std of identical values leaves float noise
        raise ValueError("zero-variance diffs; MDE is not estimable")
    z = norm.ppf(1 - alpha / 2) + norm.ppf(target_power)
    return float(z * sd / np.sqrt(len(d)))
