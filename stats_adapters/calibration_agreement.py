"""Per-dimension judge-vs-gold agreement intervals for the calibration set.

Boundary layer (knows agent-bench data formats): joins the hand-label set
against a judge/jury predictions file on (item_id, dimension), drops abstains
(score "Unknown"), and computes each dimension's headline chance-corrected
agreement with a bootstrap CI via the pure stats.agreement module.

Headline metric per dimension follows the calibration design (docs/judge-design.md):
AC1 on the prevalence-skewed groundedness and relevance dimensions where Cohen's
kappa degenerates, Cohen's kappa on the more balanced completeness gold.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stats.agreement import bootstrap_agreement_ci, cohen_kappa, gwet_ac1

_Stat = Callable[[np.ndarray, np.ndarray], float]
DIM_HEADLINE: dict[str, tuple[str, _Stat]] = {
    "groundedness": ("AC1", gwet_ac1),
    "relevance": ("AC1", gwet_ac1),
    "completeness": ("kappa", cohen_kappa),
}


@dataclass(frozen=True)
class DimensionAgreement:
    dimension: str
    metric: str
    n: int
    point: float
    ci_low: float
    ci_high: float


def _load_labels(labels_path: Path) -> dict[tuple[str, str], int | str]:
    out: dict[tuple[str, str], int | str] = {}
    for line in labels_path.read_text().splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            out[(rec["item_id"], rec["dimension"])] = rec["score"]
    return out


def paired_scores(
    labels_path: str | Path, predictions_path: str | Path
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {dimension: (gold, jury)} integer arrays, joined on (item_id,
    dimension) with abstains (score "Unknown" on either side) dropped. Mirrors
    the join in agent_bench/evaluation/calibration/report.generate_kappa_table.
    """
    labels = _load_labels(Path(labels_path))
    preds = {
        (p["item_id"], p["dimension"]): p["score"]
        for p in json.loads(Path(predictions_path).read_text())
    }
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for dim in DIM_HEADLINE:
        gold: list[int] = []
        jury: list[int] = []
        for key in sorted(k for k in labels if k[1] == dim and k in preds):
            g, j = labels[key], preds[key]
            if g == "Unknown" or j == "Unknown":
                continue
            gold.append(int(g))
            jury.append(int(j))
        out[dim] = (np.array(gold), np.array(jury))
    return out


def agreement_intervals(
    labels_path: str | Path, predictions_path: str | Path, seed: int = 20260611
) -> list[DimensionAgreement]:
    """Per-dimension headline agreement plus a 95 percent bootstrap CI."""
    pairs = paired_scores(labels_path, predictions_path)
    results = []
    for dim, (metric, stat) in DIM_HEADLINE.items():
        gold, jury = pairs[dim]
        lo, hi = bootstrap_agreement_ci(stat, gold, jury, seed=seed)
        results.append(DimensionAgreement(dim, metric, len(gold), stat(gold, jury), lo, hi))
    return results
