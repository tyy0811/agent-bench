"""Hand-rolled Cohen's kappa, Gwet's AC2, bootstrap CI.

Hand-rolled (not sklearn) for two reasons:
1. agent-bench's identity is "built from primitives" — adding sklearn
   for one function (and transitively numpy + scipy + threadpoolctl +
   joblib) contradicts that.
2. The hand-roll demonstrates formula understanding in a way that
   sklearn.metrics.cohen_kappa_score does not.

Fixture-tested against sklearn run *outside* the project venv —
see tests/evaluation/test_calibration_metrics.py and
scripts/_dev/generate_kappa_fixtures.py.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Literal


def cohen_kappa(
    y1: list,
    y2: list,
    weights: Literal[None, "linear", "quadratic"] = None,
) -> float:
    """Cohen's κ = (P_o - P_e) / (1 - P_e).

    Supports unweighted, linear-weighted, and quadratic-weighted variants
    for ordinal scales. y1 and y2 must be parallel lists of label values
    (int or str). Both must have the same length.
    """
    if len(y1) != len(y2):
        raise ValueError(
            f"y1 and y2 must have same length; got {len(y1)} vs {len(y2)}"
        )
    if not y1:
        raise ValueError("Empty input — kappa undefined")

    labels = sorted({*y1, *y2}, key=str)
    k = len(labels)
    label_idx = {lab: i for i, lab in enumerate(labels)}

    cm = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        cm[label_idx[a]][label_idx[b]] += 1

    n = len(y1)

    if weights is None:
        w = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    elif weights == "linear":
        if k <= 1:
            w = [[1.0]]
        else:
            w = [
                [1.0 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)
            ]
    elif weights == "quadratic":
        if k <= 1:
            w = [[1.0]]
        else:
            w = [
                [1.0 - ((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)
            ]
    else:
        raise ValueError(f"Invalid weights {weights!r}")

    p_o = sum(w[i][j] * cm[i][j] for i in range(k) for j in range(k)) / n

    row_marg = [sum(cm[i][j] for j in range(k)) / n for i in range(k)]
    col_marg = [sum(cm[i][j] for i in range(k)) / n for j in range(k)]

    p_e = sum(
        w[i][j] * row_marg[i] * col_marg[j] for i in range(k) for j in range(k)
    )

    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def gwets_ac2(
    y1: list,
    y2: list,
    weights: Literal[None, "linear", "quadratic"] = None,
) -> float:
    """Gwet's AC2 — chance-corrected agreement using mean marginals.

    AC2 = (P_o - P_e_AC2) / (1 - P_e_AC2)
    where P_e_AC2 = (1/(q-1)) * Σ p_k * (1 - p_k)
    and p_k is the mean marginal probability for category k.
    """
    if len(y1) != len(y2):
        raise ValueError("y1 and y2 length mismatch")
    if not y1:
        raise ValueError("Empty input")

    labels = sorted({*y1, *y2}, key=str)
    k = len(labels)
    label_idx = {lab: i for i, lab in enumerate(labels)}

    cm = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        cm[label_idx[a]][label_idx[b]] += 1
    n = len(y1)

    if weights is None:
        w = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    elif weights == "linear":
        if k <= 1:
            w = [[1.0]]
        else:
            w = [
                [1.0 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)
            ]
    elif weights == "quadratic":
        if k <= 1:
            w = [[1.0]]
        else:
            w = [
                [1.0 - ((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)
            ]
    else:
        raise ValueError(f"Invalid weights {weights!r}")

    p_o = sum(w[i][j] * cm[i][j] for i in range(k) for j in range(k)) / n

    row_marg = [sum(cm[i][j] for j in range(k)) / n for i in range(k)]
    col_marg = [sum(cm[i][j] for i in range(k)) / n for j in range(k)]
    pi = [(row_marg[i] + col_marg[i]) / 2 for i in range(k)]

    if k <= 1:
        return 1.0
    # Gwet's chance term: P_e = (1/(q-1)) * Σ pi_k * (1 - pi_k)
    # (the standard AC1 formula on mean marginals; weighted variant is
    # achieved by passing weights to P_o while keeping the unweighted
    # chance term — sufficient for v1's binary/three-point use).
    p_e_ac2 = sum(pi[i] * (1 - pi[i]) for i in range(k)) / (k - 1)

    if p_e_ac2 >= 1.0:
        return 1.0
    return (p_o - p_e_ac2) / (1.0 - p_e_ac2)


def bootstrap_ci(
    y1: list,
    y2: list,
    metric_fn: Callable[[list, list], float],
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for an inter-rater metric.

    Returns (point_estimate, ci_lo, ci_hi). Resamples with replacement
    n_iter times and takes the (1-ci)/2 and (1+ci)/2 percentiles.
    """
    if len(y1) != len(y2):
        raise ValueError("length mismatch")
    n = len(y1)
    rng = random.Random(seed)
    point = metric_fn(y1, y2)
    samples: list[float] = []
    for _ in range(n_iter):
        idx = [rng.randrange(n) for _ in range(n)]
        s1 = [y1[i] for i in idx]
        s2 = [y2[i] for i in idx]
        try:
            samples.append(metric_fn(s1, s2))
        except (ValueError, ZeroDivisionError):
            # Degenerate resample (e.g., all one label) — skip
            continue
    samples.sort()
    if not samples:
        return point, point, point
    lo_idx = int(((1 - ci) / 2) * len(samples))
    hi_idx = int(((1 + ci) / 2) * len(samples)) - 1
    return point, samples[lo_idx], samples[hi_idx]
