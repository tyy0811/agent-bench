"""Detector efficiency for canary injection (LIGO-style: inject known-bad
answers, measure the fraction the judge catches).

Per dimension this reports detection efficiency (sensitivity on the canaries
where that dimension's defect was planted) PAIRED WITH the false-positive rate
(on the canaries left clean on that dimension). Sensitivity alone is incomplete
by the detector-efficiency convention the design invokes: a judge that flags
everything scores 100 percent detection efficiency while being useless, so the
false-positive rate on clean instances is the background that makes the
efficiency meaningful. Abstains count as not-flagged in both rates and the
abstain rate is reported separately, so a judge missing defects by abstaining is
distinguishable from one confidently passing.

Pure module: pandas + the project's exact binomial interval only.
"""

from dataclasses import dataclass

import pandas as pd

from stats.intervals import clopper_pearson

_NAN = float("nan")


def flag_failing(score: int | str, top_anchor: int, *, abstained: bool) -> bool:
    """The result-blind production flag rule for one (canary, dimension) verdict.

    A verdict is flagged failing when the judge did NOT abstain and scored
    strictly below the dimension's top anchor (the rubric's passing/max level:
    1 for the binary dimensions, 2 for the three-point ones). For a three-point
    dimension a partial score is below the ceiling, so it flags failing too, not
    only the floor. The rule reads only the score against the rubric ceiling,
    never the planted-defect ground truth, so the detection table measures the
    production detector rather than an oracle. An abstain is never a flag; it is
    accounted for separately in the abstain rate.
    """
    if abstained:
        return False
    return int(score) < top_anchor


@dataclass(frozen=True)
class DimensionDetection:
    dimension: str
    # Sensitivity: of canaries where this dimension's defect was planted
    # (expected True), how many the judge flagged failing.
    n_planted: int
    detected: int
    detection_efficiency: float
    detection_ci: tuple[float, float]
    # Background: of canaries clean on this dimension (expected False), how many
    # the judge wrongly flagged failing.
    n_clean: int
    false_positives: int
    false_positive_rate: float
    fpr_ci: tuple[float, float]
    abstain_rate: float


def detection_by_dimension(df: pd.DataFrame) -> list[DimensionDetection]:
    """One DimensionDetection per dimension. ``df`` has one row per
    (canary, dimension) with columns: ``dimension``, ``expected`` (bool, the
    defect was planted on this dimension), ``flagged_failing`` (bool, the judge
    scored it failing; must already be False where the judge abstained), and
    ``abstained`` (bool). 95 percent intervals are exact Clopper-Pearson because
    the per-dimension counts are small.
    """
    out = []
    for dim in sorted(df["dimension"].unique()):
        sub = df[df["dimension"] == dim]
        planted = sub[sub["expected"]]
        clean = sub[~sub["expected"]]
        n_planted, detected = len(planted), int(planted["flagged_failing"].sum())
        n_clean, fp = len(clean), int(clean["flagged_failing"].sum())
        out.append(
            DimensionDetection(
                dimension=str(dim),
                n_planted=n_planted,
                detected=detected,
                detection_efficiency=detected / n_planted if n_planted else _NAN,
                detection_ci=clopper_pearson(detected, n_planted) if n_planted else (_NAN, _NAN),
                n_clean=n_clean,
                false_positives=fp,
                false_positive_rate=fp / n_clean if n_clean else _NAN,
                fpr_ci=clopper_pearson(fp, n_clean) if n_clean else (_NAN, _NAN),
                abstain_rate=float(sub["abstained"].mean()),
            )
        )
    return out
