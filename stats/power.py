"""Simulation-based power and MDE from measured per-question differences.

The simulation resamples the observed differences (preserving their measured
variance), shifts by delta, and tests at alpha with a paired t-test; the
closed-form normal approximation is the cross-check, per the v3.1 plan WP3.

Pure module: stdlib + numpy + scipy only (guardrail 1).
"""

import numpy as np
from scipy.stats import norm, ttest_1samp

DEFAULT_SEED = 20260611
DEFAULT_N_SIM = 2_000


def simulated_power(
    diffs: np.ndarray,
    delta: float,
    alpha: float = 0.05,
    n_sim: int = DEFAULT_N_SIM,
    seed: int = DEFAULT_SEED,
) -> float:
    centered = np.asarray(diffs, dtype=float)
    centered = centered - centered.mean()
    rng = np.random.default_rng(seed)
    n = len(centered)
    hits = 0
    for _ in range(n_sim):
        sample = rng.choice(centered, size=n, replace=True) + delta
        if ttest_1samp(sample, 0.0).pvalue < alpha:
            hits += 1
    return hits / n_sim


def mde(
    diffs: np.ndarray,
    target_power: float = 0.80,
    alpha: float = 0.05,
    n_sim: int = DEFAULT_N_SIM,
    seed: int = DEFAULT_SEED,
    tol: float = 1e-3,
) -> float:
    lo, hi = 0.0, float(np.std(diffs, ddof=1)) * 4 + 1e-6
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if simulated_power(diffs, mid, alpha=alpha, n_sim=n_sim, seed=seed) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi


def mde_normal_approx(diffs: np.ndarray, target_power: float = 0.80, alpha: float = 0.05) -> float:
    d = np.asarray(diffs, dtype=float)
    sd = float(d.std(ddof=1))
    z = norm.ppf(1 - alpha / 2) + norm.ppf(target_power)
    return float(z * sd / np.sqrt(len(d)))
