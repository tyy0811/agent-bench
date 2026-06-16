"""pass^k from epoch data: a question passes only if every epoch passes."""

from dataclasses import dataclass

import pandas as pd

from stats.intervals import clopper_pearson


@dataclass(frozen=True)
class PassK:
    k: int
    n_questions: int
    single_run_rate: float
    rate: float
    ci_low: float
    ci_high: float


def pass_k(df: pd.DataFrame, metric: str) -> PassK:
    sub = df[df["metric"] == metric]
    counts = sub.groupby("question_id")["epoch"].nunique()
    if counts.nunique() != 1:
        raise ValueError("unbalanced epochs per question")
    k = int(counts.iloc[0])
    # Single-run accuracy = expected accuracy of any one epoch (mean over all
    # question-epoch cells); reported alongside pass^k so the reader can see
    # where epoch variance bites (pass^k < single-run) and where it does not.
    single_run_rate = float(sub["score"].mean())
    all_pass = sub.groupby("question_id")["score"].min() >= 1.0
    n = len(all_pass)
    passed = int(all_pass.sum())
    # Clopper-Pearson (exact), not the plan's Wilson: the effective n is only the
    # out-of-scope questions (n=5 on FastAPI), where Wilson's normal approximation
    # is anti-conservative. CP is exact, already used for the citation zero-failure
    # bounds (no new dependency), and can only widen the interval, never narrow it
    # misleadingly. Plan deviation recorded in the WP7.2 PR.
    lo, hi = clopper_pearson(passed, n)
    return PassK(k, n, single_run_rate, passed / n, lo, hi)
