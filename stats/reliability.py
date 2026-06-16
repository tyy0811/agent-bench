"""pass^k from epoch data: a question passes only if every epoch passes."""

from dataclasses import dataclass

import pandas as pd

from stats.intervals import wilson


@dataclass(frozen=True)
class PassK:
    k: int
    n_questions: int
    rate: float
    ci_low: float
    ci_high: float


def pass_k(df: pd.DataFrame, metric: str) -> PassK:
    sub = df[df["metric"] == metric]
    counts = sub.groupby("question_id")["epoch"].nunique()
    if counts.nunique() != 1:
        raise ValueError("unbalanced epochs per question")
    k = int(counts.iloc[0])
    all_pass = sub.groupby("question_id")["score"].min() >= 1.0
    n = len(all_pass)
    passed = int(all_pass.sum())
    lo, hi = wilson(passed, n)
    return PassK(k, n, passed / n, lo, hi)
