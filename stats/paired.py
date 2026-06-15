"""Paired bootstrap CIs on per-question differences; exact McNemar.

Resampling unit per design spec section 2.5: clusters of paired differences
on a clustered-primary corpus, questions otherwise.

Pure module: stdlib + numpy + scipy only (guardrail 1).
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

DEFAULT_SEED = 20260611
DEFAULT_N_BOOT = 10_000


@dataclass(frozen=True)
class PairedResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    n_units: int


def paired_bootstrap(
    diffs: np.ndarray,
    clusters: np.ndarray | None = None,
    confidence: float = 0.90,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> PairedResult:
    diffs = np.asarray(diffs, dtype=float)
    if clusters is None:
        groups = [np.array([d]) for d in diffs]
    else:
        clusters = np.asarray(clusters)
        groups = [diffs[clusters == c] for c in np.unique(clusters)]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.integers(0, len(groups), size=len(groups))
        boot[i] = np.concatenate([groups[j] for j in picks]).mean()
    alpha = 1 - confidence
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return PairedResult(float(diffs.mean()), float(lo), float(hi), len(groups))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * binom.cdf(k, n, 0.5)
    return float(min(p, 1.0))
