"""Variance decomposition reference: hand-computed one-way random-effects
ANOVA on 3 questions x 2 epochs, scores [[0,1],[1,1],[0,0]].
Arithmetic: question means 0.5, 1.0, 0.0; grand mean 0.5.
SSB = 2*((0)^2 + (0.5)^2 + (0.5)^2) = 1.0, df=2, MSB = 0.5.
SSW = (0.25 + 0.25) + 0 + 0 = 0.5, df = 3, MSW = 1/6.
between = (MSB - MSW) / k = (0.5 - 1/6) / 2 = 1/6; within = MSW = 1/6;
ICC = (1/6) / (1/6 + 1/6) = 0.5. Provenance: hand computation above.
"""

import pandas as pd
import pytest

from stats import variance


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q2", "q2", "q3", "q3"],
            "epoch": [1, 2, 1, 2, 1, 2],
            "score": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        }
    )


def test_hand_worked_decomposition():
    res = variance.decompose(_table(), value_col="score", question_col="question_id")
    assert res.within_question == pytest.approx(1 / 6)
    assert res.between_question == pytest.approx(1 / 6)
    assert res.icc == pytest.approx(0.5)


def test_zero_within_when_epochs_identical():
    df = _table()
    df["score"] = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    res = variance.decompose(df, value_col="score", question_col="question_id")
    assert res.within_question == 0.0
    assert res.icc == 1.0


def test_negative_between_clamped_to_zero():
    df = _table()
    df["score"] = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]  # all variance within
    res = variance.decompose(df, value_col="score", question_col="question_id")
    assert res.between_question == 0.0
