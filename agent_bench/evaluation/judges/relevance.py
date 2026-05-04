"""RelevanceJudge — three-point, reference-free."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_bench.evaluation.judges.base import (
    Judge,
    ScoreResult,
    _call_judge_with_retry,
)
from agent_bench.evaluation.judges.groundedness import _system_output_hash

if TYPE_CHECKING:
    from agent_bench.agents.orchestrator import AgentResponse
    from agent_bench.evaluation.harness import GoldenQuestion


class RelevanceJudge(Judge):
    async def score(
        self,
        item: "GoldenQuestion",
        output: "AgentResponse",
        *,
        prompt_seed: int = 0,
    ) -> ScoreResult:
        schema_clause = self._json_schema_clause('0 or 1 or 2 or "Unknown"')
        prompt = (
            f"{self.rubric.render_prompt(level_permutation_seed=prompt_seed)}\n\n"
            f"---\n\n"
            f"## Question\n{item.question}\n\n"
            f"## Answer to score\n{output.answer}\n\n"
            f"Score this answer against the rubric above. Respond with ONLY a "
            f"{schema_clause}"
        )
        return await _call_judge_with_retry(
            provider=self.judge_provider,
            prompt=prompt,
            valid_scores={0, 1, 2},
            judge_id=self.judge_id,
            rubric_version=self.rubric.source_hash,
            prompt_seed=prompt_seed,
            system_output_hash=_system_output_hash(
                item.id, output.answer, [s.source for s in output.sources]
            ),
            item_id=item.id,
            abstain_allowed=self.effective_abstain_allowed,
        )
