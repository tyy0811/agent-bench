"""Canary detection adapter: ground-truth + judge-verdict join, the flag rule,
the relevance gap, and the rendered report.

The boundary builds a (canary x dimension) long frame from two plain-dict
inputs -- the canary set (per-dimension planted-defect ground truth) and the
judge predictions (per-dimension score / abstain) -- then hands it to the pure
stats core. Numbers below are hand-worked so a regression in the join or the
flag rule fails here rather than silently shifting the detection table.
"""

import math

import pandas as pd
import pytest

from stats import detection
from stats_adapters import canary

# Two canaries: c1 plants an ungrounded defect (groundedness only), c2 plants an
# incomplete defect (completeness only). Every canary carries a ground-truth
# label for all four dimensions; the non-target dimensions are the clean
# background for the false-positive rate.
_CANARIES = [
    {
        "id": "c1",
        "injection_type": "ungrounded",
        "expected_failing": {
            "groundedness": True,
            "completeness": False,
            "relevance": False,
            "citation_faithfulness": False,
        },
    },
    {
        "id": "c2",
        "injection_type": "incomplete",
        "expected_failing": {
            "groundedness": False,
            "completeness": True,
            "relevance": False,
            "citation_faithfulness": False,
        },
    },
]


# Judge verdicts. score is an int level or the "Unknown" abstain sentinel.
def _pred(item_id: str, dimension: str, score: object) -> dict:
    return {"item_id": item_id, "dimension": dimension, "score": score}


_PREDICTIONS = [
    _pred("c1", "groundedness", 0),  # below ceiling 1 -> flagged; planted -> detected
    _pred("c1", "completeness", 2),  # at ceiling 2 -> passing; clean -> correct
    _pred("c1", "relevance", 2),  # passing; clean
    _pred("c1", "citation_faithfulness", 1),  # passing; clean
    _pred("c2", "groundedness", 1),  # passing; clean
    _pred("c2", "completeness", 1),  # partial 1 < ceiling 2 -> flagged; planted -> detected
    _pred("c2", "relevance", "Unknown"),  # abstain -> never flagged
    _pred("c2", "citation_faithfulness", 1),  # passing; clean
]


def _row(df: pd.DataFrame, canary_id: str, dimension: str) -> pd.Series:
    sub = df[(df["canary_id"] == canary_id) & (df["dimension"] == dimension)]
    assert len(sub) == 1, f"expected exactly one row for {canary_id}/{dimension}"
    return sub.iloc[0]


def test_frame_has_one_row_per_canary_and_dimension():
    df = canary.build_detection_frame(_CANARIES, _PREDICTIONS)
    assert len(df) == 2 * 4  # two canaries x four dimensions
    assert set(df["dimension"]) == set(canary.TOP_ANCHOR_BY_DIMENSION)


def test_flag_rule_applied_per_dimension_ceiling():
    df = canary.build_detection_frame(_CANARIES, _PREDICTIONS)
    # Below the binary ceiling -> flagged failing; planted defect.
    g = _row(df, "c1", "groundedness")
    assert g["expected"] and g["flagged_failing"] and not g["abstained"]
    # Partial credit on a three-point dimension is still below the ceiling.
    comp = _row(df, "c2", "completeness")
    assert comp["expected"] and comp["flagged_failing"]
    # At the ceiling -> passing, even though it is a clean (non-planted) cell.
    assert not _row(df, "c1", "completeness")["flagged_failing"]


def test_abstain_derived_from_unknown_and_never_flagged():
    df = canary.build_detection_frame(_CANARIES, _PREDICTIONS)
    rel = _row(df, "c2", "relevance")
    assert rel["abstained"] is True or bool(rel["abstained"]) is True
    assert not rel["flagged_failing"]


def test_detection_numbers_hand_worked_including_relevance_gap():
    df = canary.build_detection_frame(_CANARIES, _PREDICTIONS)
    by_dim = {d.dimension: d for d in detection.detection_by_dimension(df)}

    g = by_dim["groundedness"]
    assert g.n_planted == 1 and g.detected == 1
    assert g.detection_efficiency == pytest.approx(1.0)
    assert g.n_clean == 1 and g.false_positives == 0
    assert g.false_positive_rate == pytest.approx(0.0)

    # Relevance has no planted defect: efficiency is not estimable (NaN), but it
    # still contributes a clean-background false-positive rate, and c2 abstained.
    rel = by_dim["relevance"]
    assert rel.n_planted == 0
    assert math.isnan(rel.detection_efficiency)
    assert rel.n_clean == 2 and rel.false_positives == 0
    assert rel.abstain_rate == pytest.approx(0.5)


def test_missing_prediction_is_a_loud_error():
    incomplete = [
        p for p in _PREDICTIONS if not (p["item_id"] == "c2" and p["dimension"] == "relevance")
    ]
    with pytest.raises((KeyError, ValueError)):
        canary.build_detection_frame(_CANARIES, incomplete)


def _with_score(item_id: str, dimension: str, score: object) -> list[dict]:
    """_PREDICTIONS with one (canary, dimension) verdict overridden."""
    return [
        _pred(p["item_id"], p["dimension"], score)
        if (p["item_id"], p["dimension"]) == (item_id, dimension)
        else p
        for p in _PREDICTIONS
    ]


def test_malformed_score_rejected_with_a_clear_message():
    # A hand-authored typo (wrong-case sentinel) must fail loud with guidance,
    # naming the canary, dimension, and offending value -- not crash deep in an
    # int() cast. Canary sets are authored by hand, so this path is reachable.
    bad = _with_score("c2", "relevance", "unknown")
    with pytest.raises(ValueError, match="score must be") as exc:
        canary.build_detection_frame(_CANARIES, bad)
    msg = str(exc.value)
    assert "relevance" in msg and "'unknown'" in msg  # names dimension + offending value


def test_out_of_range_integer_score_rejected():
    # 5 is not a valid level for the binary groundedness dimension (ceiling 1).
    bad = _with_score("c1", "groundedness", 5)
    with pytest.raises(ValueError, match="score must be"):
        canary.build_detection_frame(_CANARIES, bad)


def test_report_renders_all_dimensions_and_flags_the_relevance_gap():
    md = canary.build_report(_CANARIES, _PREDICTIONS, provenance="synthetic test fixtures")
    for dim in canary.TOP_ANCHOR_BY_DIMENSION:
        assert dim in md
    assert "synthetic test fixtures" in md
    # The relevance gap is called out, not silently rendered as a 0/0.
    assert "relevance" in md.lower()
    assert "n/a" in md.lower() or "not estimable" in md.lower()


def test_report_has_no_unicode_dashes():
    md = canary.build_report(_CANARIES, _PREDICTIONS, provenance="synthetic test fixtures")
    assert "—" not in md  # em dash
    assert "–" not in md  # en dash


def test_top_anchors_match_the_shipped_rubrics():
    # Drift guard: the per-dimension ceiling must equal the rubric's max score
    # level. If a rubric gains/loses a level, this fails before the flag rule
    # silently mis-grades. Imports agent_bench, which the boundary may do.
    from pathlib import Path

    from agent_bench.evaluation.judges.base import Rubric

    rubric_dir = Path(__file__).resolve().parents[2] / "agent_bench/evaluation/rubrics"
    for dim, anchor in canary.TOP_ANCHOR_BY_DIMENSION.items():
        rubric = Rubric.from_markdown_file(rubric_dir / f"{dim}.md")
        assert max(lvl.score for lvl in rubric.levels) == anchor, dim
