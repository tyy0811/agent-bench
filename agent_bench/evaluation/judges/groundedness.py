"""GroundednessJudge — binary, reference-based on item.source_snippets."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from agent_bench.evaluation.judges.base import (
    Judge,
    ScoreResult,
    _call_judge_with_retry,
)

if TYPE_CHECKING:
    from agent_bench.agents.orchestrator import AgentResponse
    from agent_bench.evaluation.harness import GoldenQuestion


def _system_output_hash(item_id: str, answer: str, sources: list[str]) -> str:
    sorted_sources = sorted(sources)
    canonical = f"{item_id}\x00{answer}\x00{','.join(sorted_sources)}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class GroundednessJudge(Judge):
    async def score(
        self,
        item: "GoldenQuestion",
        output: "AgentResponse",
        *,
        prompt_seed: int = 0,
    ) -> ScoreResult:
        snippets_block = "\n".join(
            f"[{i + 1}] {s}" for i, s in enumerate(item.source_snippets)
        )
        prompt = (
            f"{self.rubric.render_prompt(level_permutation_seed=prompt_seed)}\n\n"
            f"---\n\n"
            f"## Gold source snippets\n{snippets_block}\n\n"
            f"## Answer to score\n{output.answer}\n\n"
            f"Score this answer against the rubric above. Respond with ONLY a "
            f'JSON object: {{"reasoning": "...", "evidence_quotes": [...], '
            f'"score": 0 or 1 or "Unknown"}}.'
        )
        return await _call_judge_with_retry(
            provider=self.judge_provider,
            prompt=prompt,
            valid_scores={0, 1},
            judge_id=self.judge_id,
            rubric_version=self.rubric.source_hash,
            prompt_seed=prompt_seed,
            system_output_hash=_system_output_hash(
                item.id, output.answer, [s.source for s in output.sources]
            ),
            item_id=item.id,
            abstain_allowed=self.rubric.abstain_allowed,
        )
