"""Proportion intervals: Wilson, Clopper-Pearson, zero-failure bounds.

Pure module: stdlib + scipy only (guardrail 1).
"""

import math

from scipy.stats import beta, norm


def _check(successes: int, n: int) -> None:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes {successes} outside [0, {n}]")


def wilson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    _check(successes, n)
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def clopper_pearson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    _check(successes, n)
    alpha = 1 - confidence
    lo = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    hi = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return lo, hi


def zero_failure_upper(n: int, confidence: float = 0.95) -> float:
    """Exact Clopper-Pearson one-sided upper bound at zero observed failures."""
    _check(0, n)
    # float() because mypy types float**float as Any (negative base could be
    # complex); 1-confidence is in [0, 1] here, so the result is always real.
    return float(1 - (1 - confidence) ** (1 / n))


def rule_of_three(n: int) -> float:
    _check(0, n)
    return 3 / n
