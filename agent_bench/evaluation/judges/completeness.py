"""CompletenessJudge — three-point, reference-based on item.reference_answer."""

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


# v1.1.1: recency-positioned restatement of the rubric's "paraphrase
# allowed" semantics. Earned by the 3A probe (3/5 disputed items shifted
# 1→2 on gpt-4o-mini) which validated that gpt-4o-mini's directional
# downward bias on 3-point completeness was prompt-positionally
# correctable rather than model-intrinsic. The clause appears immediately
# before the score instruction so the conditioning isn't lost across the
# rubric body and the reasoning step. See DECISIONS "Plan 3A" entry.
PARAPHRASE_RECENCY_CLAUSE = (
    "Note: a paraphrase that captures the same meaning as a gold-answer "
    "point counts as covered. Score on content equivalence, not surface form."
)


class CompletenessJudge(Judge):
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
            f"## Reference answer (gold)\n{item.reference_answer}\n\n"
            f"## Answer to score\n{output.answer}\n\n"
            f"{PARAPHRASE_RECENCY_CLAUSE}\n\n"
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
