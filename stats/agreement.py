"""Chance-corrected agreement with bootstrap CIs over the item set.

kappa degenerates when marginals are extreme (returns nan when pe = 1);
AC1 stays defined, which is why both are reported (see the calibration
work and docs/judge-design.md). Pure module: numpy only.
"""

from collections.abc import Callable

import numpy as np

DEFAULT_SEED = 20260611
DEFAULT_N_BOOT = 10_000


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    po = float((a == b).mean())
    pe = float(sum((a == c).mean() * (b == c).mean() for c in cats))
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def gwet_ac1(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    po = float((a == b).mean())
    pgamma = np.array([((a == c).mean() + (b == c).mean()) / 2 for c in cats])
    q = len(cats)
    if q == 1:
        return 1.0
    pe = float((pgamma * (1 - pgamma)).sum() / (q - 1))
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def bootstrap_agreement_ci(
    stat: Callable[[np.ndarray, np.ndarray], float],
    a: np.ndarray,
    b: np.ndarray,
    confidence: float = 0.95,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    alpha = 1 - confidence
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = stat(a[idx], b[idx])
        # A degenerate resample (every drawn item in one category) is perfect
        # agreement; cohen_kappa is undefined there (returns nan), but the
        # canonical kappa_table pipeline scores it 1.0. Keep it as 1.0 rather
        # than drop it: dropping removes exactly the maximum-agreement resamples,
        # which truncates the upper tail anti-conservatively and would diverge
        # from the kappa_table CI this interval is meant to match.
        vals.append(1.0 if np.isnan(v) else v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
