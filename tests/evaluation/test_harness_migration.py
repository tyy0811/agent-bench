"""Tests for the harness migration to the new judge layer."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent_bench.agents.orchestrator import AgentResponse, SourceReference
from agent_bench.core.config import EvaluationConfig
from agent_bench.core.provider import LLMProvider
from agent_bench.core.types import CompletionResponse, TokenUsage


class TestJudgeProviderConfigPreserved:
    def test_judge_provider_field_still_exists_with_default(self):
        # Regression — the judge_provider knob must not be removed/renamed
        # (5 YAML configs reference it).
        c = EvaluationConfig()
        assert c.judge_provider == "openai"

    def test_judge_dimensions_default_is_three(self):
        c = EvaluationConfig()
        assert c.judge_dimensions == ["groundedness", "relevance", "completeness"]
        # citation_faithfulness is opt-in v1, default-on v1.1
        assert "citation_faithfulness" not in c.judge_dimensions


class TestEvalResultJudgeScores:
    def test_eval_result_no_longer_has_faithfulness_field(self):
        from agent_bench.evaluation.harness import EvalResult

        fields = EvalResult.model_fields
        assert "faithfulness" not in fields, (
            "faithfulness field should be removed in the supersession"
        )
        assert "correctness" not in fields, (
            "correctness field should be removed in the supersession"
        )
        assert "judge_scores" in fields, (
            "judge_scores: dict[str, ScoreResult] should be added"
        )


def _mk_judge_response(score: int) -> CompletionResponse:
    import json

    return CompletionResponse(
        content=json.dumps(
            {"reasoning": "r", "evidence_quotes": [], "score": score}
        ),
        tool_calls=[],
        usage=TokenUsage(input_tokens=10, output_tokens=10, estimated_cost_usd=0.0),
        provider="mock",
        model="m",
        latency_ms=1.0,
    )


class TestCompletenessGatedOnReferenceAnswer:
    """Regression: pre-supersession code gated correctness on
    `if q.reference_answer:` — the new per-dimension loop must preserve
    that gate so empty references don't burn tokens on guaranteed-noisy
    verdicts.
    """

    @pytest.mark.asyncio
    async def test_empty_reference_answer_skips_completeness_judge(self, tmp_path):
        from agent_bench.agents.orchestrator import Orchestrator
        from agent_bench.evaluation.harness import run_evaluation

        # Minimal golden item with an EMPTY reference_answer
        golden_path = tmp_path / "golden.json"
        golden_path.write_text(
            '[{"id": "q1", "question": "?", "expected_answer_keywords": [],'
            ' "expected_sources": [], "category": "retrieval",'
            ' "difficulty": "easy", "requires_calculator": false,'
            ' "reference_answer": ""}]'
        )

        # Mock orchestrator returning a fixed AgentResponse
        orch = AsyncMock(spec=Orchestrator)
        orch.run.return_value = AgentResponse(
            answer="Some answer.",
            sources=[SourceReference(source="a.md")],
            ranked_sources=["a.md"],
            source_chunks=["chunk a"],
            iterations=1,
            usage=TokenUsage(
                input_tokens=0, output_tokens=0, estimated_cost_usd=0.0
            ),
            latency_ms=0.0,
        )

        # Track calls to the judge provider
        judge_provider = AsyncMock(spec=LLMProvider)
        judge_provider.complete.return_value = _mk_judge_response(1)
        judge_provider.model = "test-model"

        results = await run_evaluation(
            orchestrator=orch,
            system_prompt="x",
            golden_path=golden_path,
            judge_provider=judge_provider,
        )

        assert len(results) == 1
        # Groundedness + relevance should run; completeness must be skipped
        # because reference_answer == ""
        assert "completeness" not in results[0].judge_scores, (
            "CompletenessJudge ran with empty reference_answer — "
            "should be gated on q.reference_answer truthiness"
        )
        assert "groundedness" in results[0].judge_scores
        assert "relevance" in results[0].judge_scores
