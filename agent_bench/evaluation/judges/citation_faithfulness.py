"""CitationFaithfulnessJudge — binary, per-(claim,citation) all-or-nothing."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import structlog

from agent_bench.evaluation.judges.base import (
    Judge,
    ScoreResult,
    _call_judge_with_retry,
)
from agent_bench.evaluation.judges.groundedness import _system_output_hash

if TYPE_CHECKING:
    from agent_bench.agents.orchestrator import AgentResponse
    from agent_bench.evaluation.harness import GoldenQuestion

logger = structlog.get_logger()

_CITATION_PATTERN = re.compile(r"\[source:\s*([^\]]+)\]")


def _extract_claims_with_citations(answer: str) -> list[tuple[str, str]]:
    """Return list of (claim_text, cited_source) pairs.

    A "claim" is the sentence (including its terminating punctuation)
    immediately preceding a [source:] citation. Prior citation tags
    inside `before` are stripped so multi-citation answers yield clean
    claim strings.
    """
    pairs: list[tuple[str, str]] = []
    for match in _CITATION_PATTERN.finditer(answer):
        cited = match.group(1).strip()
        before = answer[: match.start()]
        # Strip prior [source:...] tags so they don't pollute the claim
        before_clean = _CITATION_PATTERN.sub("", before)
        last_end = max(
            before_clean.rfind("."), before_clean.rfind("!"), before_clean.rfind("?")
        )
        if last_end >= 0:
            prev_end = max(
                before_clean.rfind(".", 0, last_end),
                before_clean.rfind("!", 0, last_end),
                before_clean.rfind("?", 0, last_end),
            )
            claim = before_clean[prev_end + 1 : last_end + 1].strip()
        else:
            claim = before_clean.strip()
        pairs.append((claim, cited))
    return pairs


class CitationFaithfulnessJudge(Judge):
    """Aggregates per-(claim, citation) judgments into one item-level
    binary ScoreResult. Per-pair detail is in evidence_quotes.

    All-or-nothing aggregation: any unfaithful citation → score 0.
    The rubric documents the rule explicitly.
    """

    async def score(
        self,
        item: "GoldenQuestion",
        output: "AgentResponse",
        *,
        prompt_seed: int = 0,
    ) -> ScoreResult:
        pairs = _extract_claims_with_citations(output.answer)
        # Map cited source name to its retrieved chunk text via output.source_chunks
        # (assumes index alignment with output.sources, matching harness
        # convention). If the same source appears multiple times in the
        # sources list with distinct chunks (legitimate when multiple
        # retrievals match the same doc), `setdefault` keeps only the first
        # — every "[source: X]" claim then evaluates against that one chunk,
        # a false-failure risk. Warn so the operator notices.
        source_names = [s.source for s in output.sources]
        if len(set(source_names)) < len(source_names):
            from collections import Counter

            duplicates = sorted(
                name for name, n in Counter(source_names).items() if n > 1
            )
            logger.warning(
                "citation_faithfulness_lossy_source_lookup",
                item_id=item.id,
                duplicate_source_names=duplicates,
                detail=(
                    "source name appears multiple times in output.sources "
                    "with distinct chunks; only the first chunk will be "
                    "associated with the name during citation evaluation."
                ),
            )
        source_to_chunk: dict[str, str] = {}
        for src_ref, chunk in zip(output.sources, output.source_chunks):
            source_to_chunk.setdefault(src_ref.source, chunk)

        per_pair_results: list[ScoreResult] = []
        sys_hash = _system_output_hash(
            item.id, output.answer, [s.source for s in output.sources]
        )

        if not pairs:
            return ScoreResult(
                reasoning="no_citations_in_answer",
                evidence_quotes=[],
                score=1,
                judge_id=self.judge_id,
                rubric_version=self.rubric.source_hash,
                prompt_seed=prompt_seed,
                system_output_hash=sys_hash,
                cost_usd=0.0,
                latency_ms=0.0,
            )

        accumulated_cost = 0.0
        accumulated_latency = 0.0
        any_unfaithful = False
        for claim, cited in pairs:
            # Empty claim → leading-citation case (e.g., answer starts with
            # "[source: a.md] ..." with no prior content). There is no claim
            # to evaluate against the chunk; the well-defined verdict is
            # vacuously faithful. Skip the API call; record a synthetic
            # ScoreResult so per-pair detail still appears in evidence_quotes.
            if not claim:
                per_pair_results.append(
                    ScoreResult(
                        reasoning="empty_claim_vacuously_faithful",
                        evidence_quotes=[],
                        score=1,
                        judge_id=self.judge_id,
                        rubric_version=self.rubric.source_hash,
                        prompt_seed=prompt_seed,
                        system_output_hash=sys_hash,
                        cost_usd=0.0,
                        latency_ms=0.0,
                    )
                )
                continue
            chunk = source_to_chunk.get(cited, "")
            schema_clause = self._json_schema_clause('0 or 1 or "Unknown"')
            prompt = (
                f"{self.rubric.render_prompt(level_permutation_seed=prompt_seed)}\n\n"
                f"---\n\n"
                f"## Claim (from agent's answer)\n{claim}\n\n"
                f"## Cited chunk content\n{chunk}\n\n"
                f"Does the cited chunk support the claim? Respond with ONLY a "
                f"{schema_clause}"
            )
            sub_result = await _call_judge_with_retry(
                provider=self.judge_provider,
                prompt=prompt,
                valid_scores={0, 1},
                judge_id=self.judge_id,
                rubric_version=self.rubric.source_hash,
                prompt_seed=prompt_seed,
                system_output_hash=sys_hash,
                item_id=f"{item.id}::{cited}",
                abstain_allowed=self.effective_abstain_allowed,
            )
            per_pair_results.append(sub_result)
            accumulated_cost += sub_result.cost_usd
            accumulated_latency += sub_result.latency_ms
            if sub_result.score == 0:
                any_unfaithful = True

        aggregate_score: int | Literal["Unknown"] = 0 if any_unfaithful else 1
        # Any sub-call abstain → propagate Unknown (consistent with strict-quorum)
        if any(r.abstained for r in per_pair_results):
            aggregate_score = "Unknown"

        return ScoreResult(
            reasoning=(
                f"all_or_nothing aggregate over {len(per_pair_results)} (claim, citation) pairs; "
                f"unfaithful={sum(1 for r in per_pair_results if r.score == 0)}, "
                f"abstained={sum(1 for r in per_pair_results if r.abstained)}"
            ),
            evidence_quotes=[r.reasoning[:120] for r in per_pair_results],
            score=aggregate_score,
            judge_id=self.judge_id,
            rubric_version=self.rubric.source_hash,
            prompt_seed=prompt_seed,
            system_output_hash=sys_hash,
            cost_usd=accumulated_cost,
            latency_ms=accumulated_latency,
        )
