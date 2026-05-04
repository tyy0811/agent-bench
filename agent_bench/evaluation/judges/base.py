"""Judge ABC, ScoreResult, Rubric, MockJudge, abstain-reason constants.

The Judge layer supersedes the continuous-scale answer_faithfulness /
answer_correctness functions in agent_bench/evaluation/metrics.py. See
docs/plans/2026-05-04-judge-layer-v1-design.md for the supersession
rationale and the six-axis comparison table.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --- Abstain-reason constants ---
#
# Failure-as-abstain ScoreResults carry a reasoning string with one of
# these prefixes. The calibration report pattern-matches against these
# constants for the four-way breakdown in the >20% abstain-rate flag.
# Genuine model abstain (rubric-allowed) uses the empty-string sentinel.

ABSTAIN_REASON_PROVIDER_EXHAUSTED = "judge_call_failed_after_retry: "
ABSTAIN_REASON_SCHEMA_PARSE = "schema_parse_failed_after_retry: "
ABSTAIN_REASON_OUT_OF_RANGE = "score_out_of_range_after_retry: "
ABSTAIN_REASON_GENUINE = ""


class ScoreResult(BaseModel):
    """One judge call's result. Self-contained provenance — no run
    metadata cross-reference needed for κ aggregation.

    Field order matters: reasoning + evidence_quotes come BEFORE score
    in both Pydantic field order and the JSON schema sent to the model,
    so the score conditions on the reasoning rather than being
    post-hoc rationalized.
    """

    # Reasoning-first ordering — load-bearing for the JSON schema
    reasoning: str
    evidence_quotes: list[str] = Field(default_factory=list)
    score: int | Literal["Unknown"]

    # Provenance
    judge_id: str
    rubric_version: str
    prompt_seed: int = 0
    system_output_hash: str

    # Operations
    cost_usd: float
    latency_ms: float

    @property
    def abstained(self) -> bool:
        return self.score == "Unknown"
