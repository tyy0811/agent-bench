"""Judge-unfolding demonstration: correct a canary top-anchor pass-rate for the
judge's measured completeness confusion, with uncertainty propagated.

Boundary layer (knows agent-bench data formats). The estimand is the corrected
top-anchor (passing) rate on the canary completeness items. The response matrix R
is estimated on the CALIBRATION completeness join (gold vs jury, reusing
``calibration_agreement.paired_scores``); it is then applied to a genuinely
separate observed distribution -- the canary completeness pass-rate -- and
checked against the canaries' known ground truth. That is the design's
"real target corpus removes the circularity": the calibration set supplies the
error model, the canaries supply an independent target with known truth.

The transfer is stated, not hidden: R is estimated where the judge ERRED
(calibration completeness off-diagonal mass ~0.154) and applied where it did NOT
(canary completeness off-diagonal mass 0.000, every planted defect detected and
every clean item passed), so the correction perturbs an already-correct target
and the honest test is whether the wide corrected interval stays consistent with
the known 0.70, not whether it rescues a biased measurement.

Everything reduces to the binary top-anchor split (pass = the rubric ceiling,
fail = below it) because the canary ground truth is binary (defect planted or
not). Pure offline join over committed files; the engine is ``stats.unfolding``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stats import unfolding
from stats_adapters.calibration_agreement import paired_scores
from stats_adapters.canary import TOP_ANCHOR_BY_DIMENSION, build_detection_frame

# The dimension the demonstration unfolds: completeness is the only dimension
# whose calibration confusion matrix carries real off-diagonal mass (groundedness
# and relevance are identity at this label set), so it is the only one where
# unfolding does measurable work. Settled against the real matrix, not before.
DEMO_DIMENSION = "completeness"


@dataclass(frozen=True)
class UnfoldingDemo:
    """The completeness unfolding demonstration as a set of measured numbers.

    ``raw_pass_rate`` is the jury observed top-anchor rate on the canaries;
    ``true_pass_rate`` is the canaries' known top-anchor rate; the corrected
    rates are both estimators with their bootstrap intervals. ``divergence`` is
    the gap between the two corrected points (the unidentifiability diagnostic).
    ``calibration_offdiag`` and ``canary_offdiag`` are the transfer caveat made
    numeric: the error regime R is estimated in versus the one it is applied to.
    """

    dimension: str
    n_calibration: int
    n_canary: int
    n_abstain: int
    raw_pass_rate: float
    true_pass_rate: float
    corrected_dagostini: float
    corrected_invert: float
    divergence: float
    ci_dagostini: tuple[float, float]
    ci_dagostini_r_only: tuple[float, float]
    ci_dagostini_sampling_only: tuple[float, float]
    ci_invert: tuple[float, float]
    dominant_source: str
    calibration_offdiag: float
    canary_offdiag: float


def _binary_top_anchor(scores: np.ndarray, anchor: int) -> np.ndarray:
    """1 where the score is at the rubric ceiling (pass), 0 below it (fail)."""
    binary: np.ndarray = (np.asarray(scores) == anchor).astype(int)
    return binary


def calibration_pairs(
    labels_path: str | Path, predictions_path: str | Path, dimension: str
) -> tuple[np.ndarray, np.ndarray]:
    """Binary (true, observed) top-anchor labels for the calibration join, the
    paired data the response matrix is estimated from. Reuses the abstain-dropping
    gold/jury join from :func:`calibration_agreement.paired_scores`.
    """
    gold, jury = paired_scores(labels_path, predictions_path)[dimension]
    anchor = TOP_ANCHOR_BY_DIMENSION[dimension]
    return _binary_top_anchor(gold, anchor), _binary_top_anchor(jury, anchor)


def canary_observed_and_true(
    canaries_path: str | Path, predictions_path: str | Path, dimension: str
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """The canary observed (jury) and known-true top-anchor count vectors
    ``[n_fail, n_pass]`` for ``dimension``, the abstain count, and the per-item
    off-diagonal mass, all from the validated join in
    :func:`canary.build_detection_frame` (which rejects duplicate, unknown,
    missing, or out-of-range verdicts) rather than a second hand-rolled join.
    Abstains are excluded from BOTH count vectors so the observed and true
    pass-rates share one population; ``n_abstain`` reports how many were dropped.
    True fail = a planted defect (``expected``); observed fail = the result-blind
    production flag (``flagged_failing``).
    """
    canaries = json.loads(Path(canaries_path).read_text())
    predictions = json.loads(Path(predictions_path).read_text())
    frame = build_detection_frame(canaries, predictions)
    sub = frame[frame["dimension"] == dimension]
    if len(sub) == 0:
        raise ValueError(
            f"no canary rows for dimension {dimension!r}; expected one of "
            f"{sorted(TOP_ANCHOR_BY_DIMENSION)}"
        )
    n_abstain = int(sub["abstained"].sum())
    scored = sub[~sub["abstained"]]
    if len(scored) == 0:
        raise ValueError(
            f"all {len(sub)} canary verdicts on dimension {dimension!r} abstained; "
            f"no scored items to unfold"
        )
    observed_fail = scored["flagged_failing"]
    true_fail = scored["expected"]
    n_obs = np.array([float(observed_fail.sum()), float((~observed_fail).sum())])
    n_true = np.array([float(true_fail.sum()), float((~true_fail).sum())])
    offdiag = float((observed_fail != true_fail).mean()) if len(scored) else float("nan")
    return n_obs, n_true, n_abstain, offdiag


def _offdiag_mass(true_labels: np.ndarray, observed_labels: np.ndarray) -> float:
    """Fraction of paired labels where observed disagrees with true (the off-
    diagonal confusion mass), the numeric form of the transfer caveat."""
    return float((np.asarray(true_labels) != np.asarray(observed_labels)).mean())


def build_demo(
    *,
    labels_path: str | Path,
    judge_predictions_path: str | Path,
    canaries_path: str | Path,
    canary_predictions_path: str | Path,
    dimension: str = DEMO_DIMENSION,
    n_boot: int = unfolding.DEFAULT_N_BOOT,
    seed: int = unfolding.DEFAULT_SEED,
) -> UnfoldingDemo:
    """Run the completeness unfolding demonstration end to end over committed
    files and return the measured numbers. Deterministic given the seed.
    """
    cal_true, cal_obs = calibration_pairs(labels_path, judge_predictions_path, dimension)
    n_obs, n_true, n_abstain, canary_offdiag = canary_observed_and_true(
        canaries_path, canary_predictions_path, dimension
    )
    dago = unfolding.unfold_with_uncertainty(
        cal_true, cal_obs, n_obs, method="dagostini", n_boot=n_boot, seed=seed
    )
    inv = unfolding.unfold_with_uncertainty(
        cal_true, cal_obs, n_obs, method="invert", n_boot=n_boot, seed=seed
    )
    # Index 1 is the top-anchor (pass) level in the binary [fail, pass] vector.
    return UnfoldingDemo(
        dimension=dimension,
        n_calibration=len(cal_true),
        n_canary=int(n_true.sum()),
        n_abstain=n_abstain,
        raw_pass_rate=float(n_obs[1] / n_obs.sum()),
        true_pass_rate=float(n_true[1] / n_true.sum()),
        corrected_dagostini=dago.point[1],
        corrected_invert=inv.point[1],
        divergence=dago.divergence,
        ci_dagostini=dago.ci_combined[1],
        ci_dagostini_r_only=dago.ci_r_only[1],
        ci_dagostini_sampling_only=dago.ci_sampling_only[1],
        ci_invert=inv.ci_combined[1],
        dominant_source=dago.dominant_source,
        calibration_offdiag=_offdiag_mass(cal_true, cal_obs),
        canary_offdiag=canary_offdiag,
    )
