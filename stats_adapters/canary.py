"""Canary detection boundary: join planted-defect ground truth with judge
verdicts, apply the result-blind flag rule, and render the detection report.

Boundary layer (guardrail 1): this package may import agent_bench, but the
canary adapter deliberately works off plain dicts -- the canary set (per-
dimension planted-defect ground truth) and the judge predictions (per-dimension
score / abstain) -- so the offline report regeneration pulls in neither
agent_bench nor the embedding stack. Only the rubric drift-guard test reaches
for the rubric loader.
"""

from __future__ import annotations

import hashlib
import json
import math

import pandas as pd

from stats import detection

# Per-dimension top anchor = the rubric's passing (max) score level. Provenance:
# agent_bench/evaluation/rubrics/{dimension}.md frontmatter and ## Score headers,
# read 2026-06-16 -- groundedness and citation_faithfulness are binary {0, 1}
# (ceiling 1); completeness and relevance are three_point {0, 1, 2} (ceiling 2).
# tests/stats/test_canary_adapter.py::test_top_anchors_match_the_shipped_rubrics
# pins this map to the rubric files so a level change cannot silently mis-grade.
TOP_ANCHOR_BY_DIMENSION: dict[str, int] = {
    "groundedness": 1,
    "completeness": 2,
    "relevance": 2,
    "citation_faithfulness": 1,
}

# The judge layer's abstain sentinel (ScoreResult.score == "Unknown"). This is
# the one accepted abstain representation: a per-(canary, dimension) score must
# be either an integer rubric level in [0, top_anchor] or exactly this string.
_ABSTAIN_SENTINEL = "Unknown"


def _is_abstain(score: object) -> bool:
    return score == _ABSTAIN_SENTINEL


def _validate_score(score: object, *, canary_id: str, dimension: str, top_anchor: int) -> None:
    """Reject a malformed score loudly, naming the offender. Canary sets are
    hand-authored, so a typo (a wrong-case sentinel, a float, an out-of-range
    level) must produce guidance here rather than a cryptic int() crash or a
    silently mis-bucketed verdict downstream.
    """
    if score == _ABSTAIN_SENTINEL:
        return
    if not isinstance(score, int) or isinstance(score, bool) or not (0 <= score <= top_anchor):
        raise ValueError(
            f"canary {canary_id!r} dimension {dimension!r}: score must be an "
            f"integer level 0-{top_anchor} or the abstain sentinel "
            f"{_ABSTAIN_SENTINEL!r}, got {score!r}"
        )


def build_detection_frame(canaries: list[dict], predictions: list[dict]) -> pd.DataFrame:
    """One row per (canary, dimension) with the columns the detection core
    consumes: ``dimension``, ``expected`` (planted-defect ground truth),
    ``flagged_failing`` (the result-blind production flag), ``abstained``, plus
    ``canary_id`` for traceability.

    ``predictions`` is the judge output, one record per canary x dimension with
    ``item_id``, ``dimension``, ``score`` (an int level or the ``Unknown``
    abstain sentinel). The prediction key set must match the canary set exactly:
    a duplicate ``(item_id, dimension)`` verdict, a verdict for a pair not in the
    canary set, a missing verdict, or a canary missing a dimension's ground-truth
    label is a loud error. A silent gap or last-wins overwrite would shift the
    detected count or the clean background without any signal.
    """
    expected_keys = {(c["id"], dim) for c in canaries for dim in TOP_ANCHOR_BY_DIMENSION}
    by_key: dict[tuple[str, str], dict] = {}
    for p in predictions:
        key = (p["item_id"], p["dimension"])
        if key in by_key:
            raise ValueError(
                f"duplicate prediction for canary {p['item_id']!r} dimension "
                f"{p['dimension']!r}; the predictions carry more than one verdict "
                f"for this cell (a concatenated or partially regenerated file)"
            )
        by_key[key] = p
    unexpected = sorted(by_key.keys() - expected_keys)
    if unexpected:
        raise ValueError(
            f"predictions reference unknown (canary, dimension) pairs not in the "
            f"canary set: {unexpected}"
        )
    rows: list[dict] = []
    for c in canaries:
        cid = c["id"]
        expected_map = c["expected_failing"]
        for dim, anchor in TOP_ANCHOR_BY_DIMENSION.items():
            if dim not in expected_map:
                raise ValueError(f"canary {cid!r} has no ground-truth label for dimension {dim!r}")
            if (cid, dim) not in by_key:
                raise ValueError(f"no judge verdict for canary {cid!r} on dimension {dim!r}")
            score = by_key[(cid, dim)]["score"]
            _validate_score(score, canary_id=cid, dimension=dim, top_anchor=anchor)
            abstained = _is_abstain(score)
            rows.append(
                {
                    "canary_id": cid,
                    "dimension": dim,
                    "expected": bool(expected_map[dim]),
                    "flagged_failing": detection.flag_failing(score, anchor, abstained=abstained),
                    "abstained": abstained,
                }
            )
    df = pd.DataFrame(rows)
    df["expected"] = df["expected"].astype(bool)
    df["flagged_failing"] = df["flagged_failing"].astype(bool)
    df["abstained"] = df["abstained"].astype(bool)
    return df


def injection_types_by_dimension(canaries: list[dict]) -> dict[str, list[str]]:
    """For each dimension, the sorted injection types that plant a defect there.
    A dimension whose list is empty is one no canary targets (the relevance gap).
    """
    out: dict[str, set[str]] = {dim: set() for dim in TOP_ANCHOR_BY_DIMENSION}
    for c in canaries:
        itype = str(c.get("injection_type", "") or "")
        for dim, planted in c["expected_failing"].items():
            if planted and dim in out and itype:
                out[dim].add(itype)
    return {dim: sorted(types) for dim, types in out.items()}


def _content_hash(canaries: list[dict], predictions: list[dict]) -> str:
    payload = json.dumps(
        {"canaries": canaries, "predictions": predictions},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _fmt_rate(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def _fmt_ci(ci: tuple[float, float]) -> str:
    lo, hi = ci
    if math.isnan(lo) or math.isnan(hi):
        return "n/a"
    return f"[{lo:.3f}, {hi:.3f}]"


def render_canary_report(
    results: list[detection.DimensionDetection],
    *,
    n_canaries: int,
    injection_types: dict[str, list[str]],
    content_hash: str,
    provenance: str,
) -> str:
    """Render the per-dimension detection table to markdown. Pure function of its
    inputs (no wall clock), so the committed report is byte-stable and regen is a
    no-op when the inputs are unchanged. Plain hyphens only -- no em/en dashes.
    """
    lines = ["# Canary detection report", "", f"Provenance: {provenance}.", ""]
    lines.append(
        "Each canary injects a known defect on one or more dimensions; the "
        "remaining dimensions are left clean and form the false-positive "
        "background. A verdict is flagged failing when the judge did not abstain "
        "and scored below the dimension's top anchor -- the result-blind "
        "production rule, which reads the score against the rubric ceiling, never "
        "the planted-defect label. Detection efficiency is the flagged fraction "
        "among planted defects; the false-positive rate is the flagged fraction "
        "among clean cells; both carry exact Clopper-Pearson 95 percent "
        "intervals because the per-dimension counts are small. Abstains are never "
        "detections and are reported separately, so a judge that misses defects "
        "by abstaining is distinguishable from one that confidently passes them."
    )
    lines.append("")
    lines.append(
        "| dimension | injection type(s) | planted | detected "
        "| detection efficiency | 95 percent interval | clean | false positives "
        "| false-positive rate | 95 percent interval | abstain rate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    gap_dims: list[str] = []
    for d in results:
        if d.n_planted == 0:
            gap_dims.append(d.dimension)
        itypes = ", ".join(injection_types.get(d.dimension, [])) or "none"
        lines.append(
            f"| {d.dimension} | {itypes} | {d.n_planted} | {d.detected} "
            f"| {_fmt_rate(d.detection_efficiency)} | {_fmt_ci(d.detection_ci)} "
            f"| {d.n_clean} | {d.false_positives} "
            f"| {_fmt_rate(d.false_positive_rate)} | {_fmt_ci(d.fpr_ci)} "
            f"| {d.abstain_rate:.3f} |"
        )
    lines.append("")
    for dim in gap_dims:
        lines.append(
            f"Detection gap: no canary plants a defect on the {dim} dimension, so "
            f"its detection efficiency is not estimable (n/a); the false-positive "
            f"rate is still measured on all {n_canaries} canaries as a clean-"
            f"background control."
        )
    if gap_dims:
        lines.append("")
    lines.append(f"Canaries: {n_canaries}. Input content hash: {content_hash}.")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_report(canaries: list[dict], predictions: list[dict], *, provenance: str) -> str:
    """End-to-end offline render: frame -> detection core -> markdown."""
    frame = build_detection_frame(canaries, predictions)
    results = detection.detection_by_dimension(frame)
    return render_canary_report(
        results,
        n_canaries=len(canaries),
        injection_types=injection_types_by_dimension(canaries),
        content_hash=_content_hash(canaries, predictions),
        provenance=provenance,
    )
