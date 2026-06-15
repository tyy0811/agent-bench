"""Between-question versus within-question variance from epoch data.

Balanced one-way random-effects ANOVA estimators; between-question variance
clamped at zero. Requires equal epochs per question (the epoch runner
produces balanced data; the adapter validates this upstream).

Pure module: stdlib + pandas only (guardrail 1).
"""

from dataclasses import dataclass

import pandas as pd

from stats._validate import require_finite, require_min_units


@dataclass(frozen=True)
class VarianceDecomposition:
    between_question: float
    within_question: float
    icc: float
    n_questions: int
    epochs_per_question: int


def decompose(df: pd.DataFrame, value_col: str, question_col: str) -> VarianceDecomposition:
    require_finite(df[value_col].to_numpy(), value_col)
    counts = df.groupby(question_col)[value_col].count()
    if counts.nunique() != 1:
        raise ValueError("unbalanced epochs per question; decomposition assumes balance")
    k = int(counts.iloc[0])
    if k < 2:
        raise ValueError("need at least 2 epochs per question to decompose variance")
    means = df.groupby(question_col)[value_col].mean()
    grand = df[value_col].mean()
    q = len(means)
    # Between-question variance needs >= 2 questions (msb divides by q-1); the
    # k<2 guard above is the epoch-axis twin of this question-axis guard.
    require_min_units(q, 2, "questions")
    msb = k * float(((means - grand) ** 2).sum()) / (q - 1)
    ssw = float(((df[value_col] - df[question_col].map(means)) ** 2).sum())
    msw = ssw / (q * (k - 1))
    between = max((msb - msw) / k, 0.0)
    total = between + msw
    icc = between / total if total > 0 else 0.0
    return VarianceDecomposition(between, msw, icc, q, k)
