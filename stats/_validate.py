"""Shared input preflight for the stats core (guardrail: fail loud).

stats/ is extraction-bound: it is consumed by harnesses whose inputs the
agent-bench pipeline does not control. Degenerate inputs (too few units,
non-finite scores) make the estimators return confident-wrong answers
(se=0, equivalent=True, power=1.0) rather than missing features, which is the
one failure a statistics library must not ship silently. These helpers raise
instead, operationalizing the project's "stop if nan / verify before
interpreting" discipline as library code.

Pure module: stdlib + numpy only (guardrail 1).
"""

import numpy as np


def require_finite(values: np.ndarray, name: str = "values") -> np.ndarray:
    """Return values as float64, raising if any element is nan or inf."""
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values (nan/inf); refusing to estimate")
    return arr


def require_min_units(n: int, minimum: int, name: str = "units") -> None:
    """Raise if fewer than `minimum` independent units are available to estimate."""
    if n < minimum:
        raise ValueError(f"need at least {minimum} {name} to estimate, got {n}")
