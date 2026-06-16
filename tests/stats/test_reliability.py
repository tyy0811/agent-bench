"""pass^k for refusal behavior: fraction of questions passing in all k epochs,
with a Wilson interval over questions. Hand reference: 3 questions over 2
epochs with refusal_correct (1,1), (1,0), (1,1): pass^2 = 2/3."""

import pandas as pd
import pytest

from stats import reliability
from stats.intervals import wilson


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q2", "q2", "q3", "q3"],
            "epoch": [1, 2, 1, 2, 1, 2],
            "metric": ["refusal_correct"] * 6,
            "score": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        }
    )


def test_pass_k_hand_worked():
    res = reliability.pass_k(_table(), metric="refusal_correct")
    assert res.k == 2
    assert res.n_questions == 3
    assert res.rate == pytest.approx(2 / 3)
    assert (res.ci_low, res.ci_high) == pytest.approx(wilson(2, 3))


def test_requires_balanced_epochs():
    df = _table().drop(index=5)
    with pytest.raises(ValueError):
        reliability.pass_k(df, metric="refusal_correct")
