"""rubric_permute — runs the same judge with permuted rubric levels and aggregates."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from agent_bench.evaluation.judges.base import Judge, ScoreResult

if TYPE_CHECKING:
    from agent_bench.agents.orchestrator import AgentResponse
    from agent_bench.evaluation.harness import GoldenQuestion


def _aggregate_scores(scores: list[int], scale: str) -> int:
    """Discretize aggregated score per scale.

    Binary: threshold 0.5 with ties → 0 (conservative).
    Three-point: round to nearest with ties → lower level (conservative).
    """
    mean = sum(scores) / len(scores)
    if scale == "binary":
        return 1 if mean > 0.5 else 0
    floor = int(mean)
    frac = mean - floor
    if frac > 0.5:
        return floor + 1
    return floor


class PermutedJudge:
    """Wraps a Judge; runs N permutations with different prompt_seeds.

    Aggregation:
    - Any abstain in any permutation → aggregate score = "Unknown".
    - Otherwise, discretize the per-permutation scores per scale.

    Per-permutation ScoreResults are written to the sidecar JSONL on
    every score() call (one batch per call, append-mode JSONL across calls).
    """

    def __init__(
        self,
        judge: Judge,
        n: int = 2,
        seeds: list[int] | None = None,
        sidecar_path: Path | str | None = None,
    ) -> None:
        self.judge = judge
        self.n = n
        self.seeds = seeds if seeds is not None else list(range(1, n + 1))
        if len(self.seeds) != n:
            raise ValueError(f"seeds length {len(self.seeds)} != n {n}")
        self.sidecar_path = Path(sidecar_path) if sidecar_path else None
        self.judge_id = f"{judge.judge_id}_perm{n}"

    async def score(
        self,
        item: "GoldenQuestion",
        output: "AgentResponse",
    ) -> ScoreResult:
        per_perm_results: list[ScoreResult] = []
        for seed in self.seeds:
            r = await self.judge.score(item, output, prompt_seed=seed)
            per_perm_results.append(r)

        if self.sidecar_path is not None:
            self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            with self.sidecar_path.open("a", encoding="utf-8") as f:
                for r in per_perm_results:
                    f.write(r.model_dump_json() + "\n")

        any_abstain = any(r.abstained for r in per_perm_results)
        if any_abstain:
            score: int | Literal["Unknown"] = "Unknown"
            reasoning = (
                f"any_abstain_propagated: "
                f"{sum(1 for r in per_perm_results if r.abstained)}/{self.n} "
                f"permutations abstained"
            )
        else:
            score = _aggregate_scores(
                [int(r.score) for r in per_perm_results],
                self.judge.rubric.scale,
            )
            reasoning = (
                f"perm_mean over {self.n} seeds: {[r.score for r in per_perm_results]}"
            )

        return ScoreResult(
            reasoning=reasoning,
            evidence_quotes=[],
            score=score,
            judge_id=self.judge_id,
            rubric_version=self.judge.rubric.source_hash,
            prompt_seed=0,
            system_output_hash=per_perm_results[0].system_output_hash,
            cost_usd=sum(r.cost_usd for r in per_perm_results),
            latency_ms=sum(r.latency_ms for r in per_perm_results),
        )


def rubric_permute(
    judge: Judge,
    n: int = 2,
    seeds: list[int] | None = None,
    sidecar_path: Path | str | None = None,
) -> PermutedJudge:
    return PermutedJudge(judge=judge, n=n, seeds=seeds, sidecar_path=sidecar_path)
