"""Detector-efficiency references. Hand-worked: groundedness defect planted on 2
canaries, the judge catches 1 -> detection efficiency 1/2; left clean on 2, the
judge wrongly flags 1 -> false-positive rate 1/2. Exact Clopper-Pearson CIs.
Abstains count as not-flagged and are reported separately."""

import pandas as pd
import pytest

from stats import detection
from stats.intervals import clopper_pearson

_COLS = ["dimension", "expected", "flagged_failing", "abstained"]


def test_detection_efficiency_and_false_positive_rate_hand_worked():
    # (dimension, expected=defect planted, flagged_failing, abstained)
    rows = [
        ("groundedness", True, True, False),  # planted, detected
        ("groundedness", True, False, False),  # planted, missed
        ("groundedness", False, True, False),  # clean, false positive
        ("groundedness", False, False, False),  # clean, correct
    ]
    g = detection.detection_by_dimension(pd.DataFrame(rows, columns=_COLS))[0]
    assert g.dimension == "groundedness"
    assert g.n_planted == 2 and g.detected == 1
    assert g.detection_efficiency == pytest.approx(0.5)
    assert g.detection_ci == pytest.approx(clopper_pearson(1, 2))
    assert g.n_clean == 2 and g.false_positives == 1
    assert g.false_positive_rate == pytest.approx(0.5)
    assert g.fpr_ci == pytest.approx(clopper_pearson(1, 2))
    assert g.abstain_rate == pytest.approx(0.0)


def test_abstain_is_not_a_detection_and_is_reported_separately():
    rows = [
        ("completeness", True, False, True),  # abstain on a planted defect -> not detected
        ("completeness", True, True, False),  # detected
    ]
    d = detection.detection_by_dimension(pd.DataFrame(rows, columns=_COLS))[0]
    assert d.n_planted == 2 and d.detected == 1  # the abstain is not counted as detected
    assert d.detection_efficiency == pytest.approx(0.5)
    assert d.abstain_rate == pytest.approx(0.5)


# --- flag_failing: the result-blind production flag rule ---
# flagged_failing = (not abstained) AND (score < the dimension's top anchor).
# Top anchor is the rubric's passing (max) level: 1 for the binary dimensions,
# 2 for the three-point dimensions. The rule is result-blind: it reads only the
# judge's score against the rubric ceiling, never the ground-truth defect label.


def test_flag_failing_binary_dimension():
    # Groundedness/citation_faithfulness top anchor is 1.
    assert detection.flag_failing(0, 1, abstained=False) is True  # below ceiling -> failing
    assert detection.flag_failing(1, 1, abstained=False) is False  # at ceiling -> passing


def test_flag_failing_three_point_partial_credit_is_failing():
    # Completeness/relevance top anchor is 2; a partial score of 1 is below the
    # ceiling, so production flags it failing (not only the score-0 floor).
    assert detection.flag_failing(1, 2, abstained=False) is True
    assert detection.flag_failing(2, 2, abstained=False) is False


def test_flag_failing_abstain_is_never_a_flag():
    # An abstain ("Unknown") is not a flag regardless of the ceiling; it is
    # accounted for in the separate abstain rate, never as a detection.
    assert detection.flag_failing("Unknown", 1, abstained=True) is False
    assert detection.flag_failing("Unknown", 2, abstained=True) is False
