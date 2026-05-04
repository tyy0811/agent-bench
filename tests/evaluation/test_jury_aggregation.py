"""Tests for PermutedJudge and Jury — aggregation, quorum, sidecar."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from agent_bench.agents.orchestrator import AgentResponse, SourceReference
from agent_bench.core.provider import LLMProvider
from agent_bench.core.types import CompletionResponse, TokenUsage
from agent_bench.evaluation.harness import GoldenQuestion
from agent_bench.evaluation.judges.base import Rubric
from agent_bench.evaluation.judges.relevance import RelevanceJudge


def _mk_response(content: str) -> CompletionResponse:
    return CompletionResponse(
        content=content,
        tool_calls=[],
        usage=TokenUsage(input_tokens=10, output_tokens=10, estimated_cost_usd=0.0001),
        provider="mock",
        model="m",
        latency_ms=1.0,
    )


def _vj(score) -> str:
    return json.dumps({"reasoning": "r", "evidence_quotes": [], "score": score})


def _item(item_id: str = "i1") -> GoldenQuestion:
    return GoldenQuestion(
        id=item_id,
        question="?",
        expected_answer_keywords=[],
        expected_sources=[],
        category="retrieval",
        difficulty="easy",
        requires_calculator=False,
    )


def _output(answer: str = "A.") -> AgentResponse:
    return AgentResponse(
        answer=answer,
        sources=[SourceReference(source="x.md")],
        iterations=1,
        usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
        latency_ms=0,
    )


def _relevance_judge_with_responses(responses: list[str]) -> RelevanceJudge:
    rubric = Rubric.from_markdown_file("agent_bench/evaluation/rubrics/relevance.md")
    provider = AsyncMock(spec=LLMProvider)
    provider.complete.side_effect = [_mk_response(r) for r in responses]
    return RelevanceJudge(judge_provider=provider, rubric=rubric, model_id="m")


class TestPermutedJudge:
    @pytest.mark.asyncio
    async def test_runs_n_permutations_and_means(self, tmp_path):
        from agent_bench.evaluation.variance.rubric_permute import rubric_permute

        # Two seeds produce two scores: 1 and 2; mean=1.5; ties→lower → 1
        judge = _relevance_judge_with_responses([_vj(1), _vj(2)])
        permuted = rubric_permute(
            judge, n=2, seeds=[1, 2], sidecar_path=tmp_path / "side.jsonl"
        )
        result = await permuted.score(_item(), _output())
        assert result.score == 1
        assert result.judge_id == "m_relevance_perm2"
        assert result.prompt_seed == 0

    @pytest.mark.asyncio
    async def test_any_abstain_propagates_unknown(self, tmp_path):
        from agent_bench.evaluation.variance.rubric_permute import rubric_permute

        judge = _relevance_judge_with_responses([_vj(1), _vj("Unknown")])
        permuted = rubric_permute(
            judge, n=2, seeds=[1, 2], sidecar_path=tmp_path / "side.jsonl"
        )
        result = await permuted.score(_item(), _output())
        assert result.score == "Unknown"
        assert result.abstained

    @pytest.mark.asyncio
    async def test_writes_per_permutation_sidecar(self, tmp_path):
        from agent_bench.evaluation.variance.rubric_permute import rubric_permute

        sidecar = tmp_path / "perm_members.jsonl"
        judge = _relevance_judge_with_responses([_vj(2), _vj(2)])
        permuted = rubric_permute(judge, n=2, seeds=[5, 7], sidecar_path=sidecar)
        await permuted.score(_item(), _output())

        lines = sidecar.read_text().strip().split("\n")
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        assert {r["prompt_seed"] for r in records} == {5, 7}


class TestJury:
    @pytest.mark.asyncio
    async def test_mean_aggregation_two_judges(self, tmp_path):
        from agent_bench.evaluation.variance.jury import jury

        j1 = _relevance_judge_with_responses([_vj(2)])
        j2 = _relevance_judge_with_responses([_vj(2)])
        j1.judge_id = "claude-haiku_relevance"
        j2.judge_id = "gpt-4o-mini_relevance"

        ju = jury(
            judges=[j1, j2], aggregation="mean", sidecar_path=tmp_path / "jury.jsonl"
        )
        result = await ju.score(_item(), _output())
        assert result.score == 2
        assert result.judge_id == "jury_v1_mean"

    @pytest.mark.asyncio
    async def test_strict_quorum_default_abstains_on_one_failure(self, tmp_path):
        from agent_bench.evaluation.variance.jury import jury

        j1 = _relevance_judge_with_responses([_vj(1)])
        j1.judge_id = "claude-haiku_relevance"
        # Both attempts return garbage → abstain via schema-parse-after-retry
        j2 = _relevance_judge_with_responses(["garbage", "garbage"])
        j2.judge_id = "gpt-4o-mini_relevance"

        ju = jury(
            judges=[j1, j2], aggregation="mean", sidecar_path=tmp_path / "jury.jsonl"
        )
        result = await ju.score(_item(), _output())
        assert result.score == "Unknown"
        assert "jury_below_quorum" in result.reasoning
        assert "1/2" in result.reasoning

    @pytest.mark.asyncio
    async def test_sidecar_captures_both_members_including_abstain(self, tmp_path):
        from agent_bench.evaluation.variance.jury import jury

        j1 = _relevance_judge_with_responses([_vj(1)])
        j1.judge_id = "claude-haiku_relevance"
        j2 = _relevance_judge_with_responses(["garbage", "garbage"])
        j2.judge_id = "gpt-4o-mini_relevance"

        sidecar = tmp_path / "jury.jsonl"
        ju = jury(judges=[j1, j2], aggregation="mean", sidecar_path=sidecar)
        await ju.score(_item(), _output())

        records = [
            json.loads(line) for line in sidecar.read_text().strip().split("\n")
        ]
        assert len(records) == 2
        scores = [r["score"] for r in records]
        assert 1 in scores
        assert "Unknown" in scores

    @pytest.mark.asyncio
    async def test_kappa_weighted_requires_weights(self, tmp_path):
        from agent_bench.evaluation.variance.jury import jury

        j1 = _relevance_judge_with_responses([_vj(2)])
        with pytest.raises(ValueError, match="weights"):
            jury(judges=[j1], aggregation="kappa_weighted")

    @pytest.mark.asyncio
    async def test_cancel_on_non_retryable(self, tmp_path):
        """Non-retryable exception in any member must propagate immediately."""
        from agent_bench.evaluation.judges.base import Rubric
        from agent_bench.evaluation.variance.jury import jury

        rubric = Rubric.from_markdown_file(
            "agent_bench/evaluation/rubrics/relevance.md"
        )
        # j1 raises ValueError (caller bug — not in retryable taxonomy)
        provider1 = AsyncMock(spec=LLMProvider)
        provider1.complete.side_effect = ValueError("auth_error")
        j1 = RelevanceJudge(judge_provider=provider1, rubric=rubric, model_id="m1")

        # j2 would succeed if it ran
        provider2 = AsyncMock(spec=LLMProvider)
        provider2.complete.return_value = _mk_response(_vj(1))
        j2 = RelevanceJudge(judge_provider=provider2, rubric=rubric, model_id="m2")

        ju = jury(
            judges=[j1, j2], aggregation="mean", sidecar_path=tmp_path / "jury.jsonl"
        )
        with pytest.raises(ValueError, match="auth_error"):
            await ju.score(_item(), _output())
