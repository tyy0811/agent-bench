"""Jury — multi-judge aggregator with strict-quorum default and sidecar."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import structlog

from agent_bench.evaluation.judges.base import Judge, ScoreResult
from agent_bench.evaluation.variance.rubric_permute import _aggregate_scores

if TYPE_CHECKING:
    from agent_bench.agents.orchestrator import AgentResponse
    from agent_bench.evaluation.harness import GoldenQuestion

_DEFAULT_SIDECAR_TEMPLATE = "results/calibration_v1_judge_{aggregation}_members.jsonl"

logger = structlog.get_logger()


def _discretize_mean(mean: float, scale: str) -> int:
    """Discretize a float mean to a discrete level per scale, ties → lower
    (mirrors `_aggregate_scores`'s policy without going through int(round())
    which would invoke Python's banker's rounding and silently violate the
    tie-breaking contract).
    """
    if scale == "binary":
        return 1 if mean > 0.5 else 0
    floor = int(mean)
    frac = mean - floor
    return floor + 1 if frac > 0.5 else floor


class Jury:
    """Aggregates a list of Judge instances into one ScoreResult per item.

    Strict quorum default (quorum = len(judges)): any member abstain →
    aggregate abstain. The parameter exists in v1 so v1.1's 3-judge jury
    can shift to quorum=2 (majority) without rearchitecting failure
    semantics.

    Per-member ScoreResults always written to sidecar (successes and
    failure-as-abstains alike). Provider non-retryable exceptions in
    any member raise immediately, cancelling sibling gather tasks.
    """

    def __init__(
        self,
        judges: list[Judge],
        aggregation: Literal["mean", "kappa_weighted"],
        weights: dict[str, float] | None = None,
        quorum: int | None = None,
        sidecar_path: Path | str | None = None,
    ) -> None:
        if not judges:
            raise ValueError("jury requires at least one judge")
        if aggregation == "kappa_weighted" and not weights:
            raise ValueError(
                "kappa_weighted aggregation requires explicit weights "
                "(computed offline on calibration set; not at jury construction)"
            )
        self.judges = judges
        self.aggregation = aggregation
        self.weights = weights or {}
        self.quorum = quorum if quorum is not None else len(judges)
        self.sidecar_path = (
            Path(sidecar_path)
            if sidecar_path is not None
            else Path(_DEFAULT_SIDECAR_TEMPLATE.format(aggregation=aggregation))
        )
        self.judge_id = f"jury_v1_{aggregation}"

    async def score(
        self,
        item: "GoldenQuestion",
        output: "AgentResponse",
    ) -> ScoreResult:
        # return_exceptions=False → first exception cancels siblings
        member_results: list[ScoreResult] = await asyncio.gather(
            *[j.score(item, output) for j in self.judges],
            return_exceptions=False,
        )

        # Sidecar (append; one line per member per call)
        self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with self.sidecar_path.open("a", encoding="utf-8") as f:
            for r in member_results:
                f.write(r.model_dump_json() + "\n")

        successful = [r for r in member_results if not r.abstained]
        sys_hash = member_results[0].system_output_hash

        if len(successful) < self.quorum:
            return ScoreResult(
                reasoning=(
                    f"jury_below_quorum: {len(successful)}/{len(self.judges)} "
                    f"members succeeded; required {self.quorum}"
                ),
                evidence_quotes=[],
                score="Unknown",
                judge_id=self.judge_id,
                rubric_version=member_results[0].rubric_version,
                prompt_seed=0,
                system_output_hash=sys_hash,
                cost_usd=sum(r.cost_usd for r in member_results),
                latency_ms=max(r.latency_ms for r in member_results),
            )

        # Aggregate over successful members
        scores = [int(r.score) for r in successful]
        scale = self.judges[0].rubric.scale
        applied_weights: list[float] = []
        if self.aggregation == "mean":
            agg = _aggregate_scores(scores, scale)
        else:  # kappa_weighted
            # Weight successful members by judge_id; missing weights → 1.0
            # (mean fallback). Warn loudly when this fallback fires —
            # `kappa_weighted` is supposed to use explicit weights, and
            # silently substituting 1.0 violates that contract.
            for r in successful:
                if r.judge_id not in self.weights:
                    logger.warning(
                        "jury_missing_weight_fallback_to_one",
                        judge_id=r.judge_id,
                        aggregation=self.aggregation,
                        configured_weights=sorted(self.weights.keys()),
                    )
                applied_weights.append(self.weights.get(r.judge_id, 1.0))
            weighted_sum = sum(s * w for s, w in zip(scores, applied_weights))
            weight_total = sum(applied_weights)
            weighted_mean = (
                weighted_sum / weight_total if weight_total > 0 else 0.0
            )
            # Discretize via the shared ties-to-lower policy (NOT int(round())
            # which uses banker's rounding and would diverge from the `mean`
            # path on half-integer aggregates).
            agg = _discretize_mean(weighted_mean, scale)

        # Reasoning string reports the per-member weights actually applied
        # (not the constructor's dict — the dict may be missing entries that
        # silently fell back to 1.0; printing the constructor's dict would
        # conceal that fallback from anyone debugging a calibration row).
        weights_str = applied_weights if self.aggregation == "kappa_weighted" else "n/a"
        return ScoreResult(
            reasoning=(
                f"jury_{self.aggregation}: "
                f"members={[r.score for r in successful]}, "
                f"weights={weights_str}"
            ),
            evidence_quotes=[],
            score=agg,
            judge_id=self.judge_id,
            rubric_version=member_results[0].rubric_version,
            prompt_seed=0,
            system_output_hash=sys_hash,
            cost_usd=sum(r.cost_usd for r in member_results),
            latency_ms=max(r.latency_ms for r in member_results),
        )


def jury(
    judges: list[Judge],
    aggregation: Literal["mean", "kappa_weighted"],
    weights: dict[str, float] | None = None,
    quorum: int | None = None,
    sidecar_path: Path | str | None = None,
) -> Jury:
    return Jury(
        judges=judges,
        aggregation=aggregation,
        weights=weights,
        quorum=quorum,
        sidecar_path=sidecar_path,
    )
