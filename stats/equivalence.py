"""TOST equivalence on paired differences via the bootstrap 90 percent CI.

Pre-registered (design spec section 2.4, frozen 2026-06-11): margin 0.10
absolute on P@5 and R@5, alpha 0.05 per one-sided test, no multiplicity
adjustment across the two metrics, paired score = per-question mean over
epochs. support_margin is the smallest margin at which equivalence would
pass: the larger absolute endpoint of the 90 percent CI, reported in both
the pass and fail branches (README wording in stats/report.py).

Pure module: stdlib + numpy + scipy (via stats.paired) only (guardrail 1).
"""

from dataclasses import dataclass

import numpy as np

from stats.paired import DEFAULT_N_BOOT, DEFAULT_SEED, paired_bootstrap

DEFAULT_MARGIN = 0.10
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class TostResult:
    equivalent: bool
    ci_low: float
    ci_high: float
    margin: float
    support_margin: float
    n_units: int


def tost_paired(
    diffs: np.ndarray,
    margin: float = DEFAULT_MARGIN,
    alpha: float = DEFAULT_ALPHA,
    clusters: np.ndarray | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> TostResult:
    res = paired_bootstrap(
        np.asarray(diffs, dtype=float),
        clusters=clusters,
        confidence=1 - 2 * alpha,
        n_boot=n_boot,
        seed=seed,
    )
    equivalent = -margin < res.ci_low and res.ci_high < margin
    support = max(abs(res.ci_low), abs(res.ci_high))
    return TostResult(equivalent, res.ci_low, res.ci_high, margin, support, res.n_units)
