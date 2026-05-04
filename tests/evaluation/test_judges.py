"""Tests for Judge ABC, ScoreResult, MockJudge, abstain reasons."""

from __future__ import annotations

import pytest

from agent_bench.evaluation.judges.base import (
    ABSTAIN_REASON_GENUINE,
    ABSTAIN_REASON_OUT_OF_RANGE,
    ABSTAIN_REASON_PROVIDER_EXHAUSTED,
    ABSTAIN_REASON_SCHEMA_PARSE,
    ScoreResult,
)


class TestAbstainConstants:
    def test_genuine_is_empty_sentinel(self):
        assert ABSTAIN_REASON_GENUINE == ""

    def test_failure_prefixes_end_with_colon_space(self):
        # All non-genuine prefixes must end with ": " so f-string concatenation
        # produces a parseable "PREFIX: detail" pattern.
        for prefix in (
            ABSTAIN_REASON_PROVIDER_EXHAUSTED,
            ABSTAIN_REASON_SCHEMA_PARSE,
            ABSTAIN_REASON_OUT_OF_RANGE,
        ):
            assert prefix.endswith(": "), f"Bad prefix: {prefix!r}"
            assert "_" in prefix.rstrip(": "), f"Prefix should be snake_case: {prefix!r}"


class TestScoreResult:
    def _base_kwargs(self) -> dict:
        return {
            "reasoning": "test",
            "evidence_quotes": [],
            "judge_id": "mock_groundedness",
            "rubric_version": "abc123",
            "system_output_hash": "def456",
            "cost_usd": 0.001,
            "latency_ms": 100.0,
        }

    def test_int_score_valid(self):
        r = ScoreResult(score=1, **self._base_kwargs())
        assert r.score == 1
        assert r.abstained is False

    def test_unknown_score_is_abstain(self):
        r = ScoreResult(score="Unknown", **self._base_kwargs())
        assert r.score == "Unknown"
        assert r.abstained is True

    def test_field_order_reasoning_first(self):
        # The JSON schema sent to the model puts reasoning before score.
        # Pydantic field order in model_fields drives JSON schema order.
        fields = list(ScoreResult.model_fields.keys())
        assert fields.index("reasoning") < fields.index("score"), (
            f"reasoning must come before score; got order: {fields}"
        )
        assert fields.index("evidence_quotes") < fields.index("score"), (
            f"evidence_quotes must come before score; got order: {fields}"
        )

    def test_prompt_seed_defaults_to_zero(self):
        r = ScoreResult(score=0, **self._base_kwargs())
        assert r.prompt_seed == 0

    def test_score_rejects_other_strings(self):
        with pytest.raises(ValueError):
            ScoreResult(score="maybe", **self._base_kwargs())  # type: ignore[arg-type]


from abc import ABC
from pathlib import Path

from agent_bench.evaluation.judges.base import Judge


class TestJudgeABC:
    def test_judge_is_abstract(self):
        assert issubclass(Judge, ABC)
        # Cannot instantiate directly — score is abstract
        with pytest.raises(TypeError, match="abstract"):
            Judge(judge_provider=None, rubric=None, model_id="test")  # type: ignore[abstract,arg-type]

    def test_judge_id_built_from_model_and_dimension(self):
        # Concrete subclass that satisfies the abstract method
        class _ConcreteJudge(Judge):
            async def score(self, item, output, *, prompt_seed=0):
                raise NotImplementedError

        from agent_bench.evaluation.judges.base import Rubric

        rubric = Rubric.from_markdown_file(
            Path(__file__).parent / "fixtures" / "rubrics_valid_binary.md"
        )
        j = _ConcreteJudge(judge_provider=None, rubric=rubric, model_id="claude-haiku-4-5")  # type: ignore[arg-type]
        assert j.judge_id == "claude-haiku-4-5_groundedness"


from agent_bench.evaluation.judges.base import MockJudge


class TestMockJudge:
    def _verdict(self, item_id: str, score: int = 1) -> ScoreResult:
        return ScoreResult(
            reasoning=f"prebaked for {item_id}",
            evidence_quotes=[],
            score=score,
            judge_id="mock_groundedness",
            rubric_version="abc",
            system_output_hash="def",
            cost_usd=0.0,
            latency_ms=0.0,
        )

    @pytest.mark.asyncio
    async def test_returns_prebaked_verdict(self, monkeypatch):
        from agent_bench.agents.orchestrator import AgentResponse, SourceReference
        from agent_bench.core.types import TokenUsage
        from agent_bench.evaluation.harness import GoldenQuestion

        verdict = self._verdict("item_001", score=1)
        mj = MockJudge(verdicts={"item_001": verdict})

        item = GoldenQuestion(
            id="item_001",
            question="?",
            expected_answer_keywords=[],
            expected_sources=[],
            category="retrieval",
            difficulty="easy",
            requires_calculator=False,
        )
        output = AgentResponse(
            answer="x",
            sources=[SourceReference(source="a.md")],
            iterations=1,
            usage=TokenUsage(
                input_tokens=0, output_tokens=0, estimated_cost_usd=0
            ),
            latency_ms=0,
        )
        result = await mj.score(item, output)
        assert result.score == 1
        assert result.reasoning == "prebaked for item_001"

    @pytest.mark.asyncio
    async def test_raises_lookuperror_on_missing_key(self):
        from agent_bench.agents.orchestrator import AgentResponse
        from agent_bench.core.types import TokenUsage
        from agent_bench.evaluation.harness import GoldenQuestion

        mj = MockJudge(verdicts={"item_001": self._verdict("item_001")})

        item = GoldenQuestion(
            id="item_999_NOT_PRESENT",
            question="?",
            expected_answer_keywords=[],
            expected_sources=[],
            category="retrieval",
            difficulty="easy",
            requires_calculator=False,
        )
        output = AgentResponse(
            answer="x",
            sources=[],
            iterations=1,
            usage=TokenUsage(
                input_tokens=0, output_tokens=0, estimated_cost_usd=0
            ),
            latency_ms=0,
        )
        with pytest.raises(LookupError, match="item_999_NOT_PRESENT"):
            await mj.score(item, output)


import json
from unittest.mock import AsyncMock

from agent_bench.core.provider import (
    LLMProvider,
    ProviderRateLimitError,
)
from agent_bench.core.types import CompletionResponse, TokenUsage
from agent_bench.evaluation.judges.base import _call_judge_with_retry


def _mk_response(content: str) -> CompletionResponse:
    return CompletionResponse(
        content=content,
        tool_calls=[],
        usage=TokenUsage(input_tokens=10, output_tokens=10, estimated_cost_usd=0.0001),
        provider="mock",
        model="mock-1",
        latency_ms=1.0,
    )


def _valid_json(score: int) -> str:
    return json.dumps(
        {
            "reasoning": "test reasoning",
            "evidence_quotes": ["q1"],
            "score": score,
        }
    )


class TestCallJudgeWithRetry:
    @pytest.mark.asyncio
    async def test_first_attempt_success(self):
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.return_value = _mk_response(_valid_json(1))

        result = await _call_judge_with_retry(
            provider=provider,
            prompt="test prompt",
            valid_scores={0, 1},
            judge_id="claude-haiku-4-5_groundedness",
            rubric_version="abc",
            prompt_seed=0,
            system_output_hash="def",
            item_id="item_001",
        )
        assert result.score == 1
        assert provider.complete.await_count == 1

    @pytest.mark.asyncio
    async def test_schema_parse_then_retry_success(self):
        import structlog

        provider = AsyncMock(spec=LLMProvider)
        provider.complete.side_effect = [
            _mk_response("not json at all"),
            _mk_response(_valid_json(0)),
        ]

        with structlog.testing.capture_logs() as logs:
            result = await _call_judge_with_retry(
                provider=provider,
                prompt="test prompt",
                valid_scores={0, 1},
                judge_id="claude-haiku-4-5_groundedness",
                rubric_version="abc",
                prompt_seed=0,
                system_output_hash="def",
                item_id="item_001",
            )
        assert result.score == 0
        assert provider.complete.await_count == 2
        # First-attempt-failure log must have fired even though retry succeeded
        assert any(
            entry.get("event") == "judge_first_attempt_failure" for entry in logs
        ), f"no judge_first_attempt_failure log in {logs!r}"

    @pytest.mark.asyncio
    async def test_schema_parse_twice_abstains_with_prefix(self):
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.side_effect = [
            _mk_response("garbage"),
            _mk_response("also garbage"),
        ]

        result = await _call_judge_with_retry(
            provider=provider,
            prompt="test prompt",
            valid_scores={0, 1},
            judge_id="claude-haiku-4-5_groundedness",
            rubric_version="abc",
            prompt_seed=0,
            system_output_hash="def",
            item_id="item_001",
        )
        assert result.abstained
        assert result.reasoning.startswith(ABSTAIN_REASON_SCHEMA_PARSE)

    @pytest.mark.asyncio
    async def test_score_out_of_range_twice_abstains_with_prefix(self):
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.side_effect = [
            _mk_response(_valid_json(5)),
            _mk_response(_valid_json(7)),
        ]

        result = await _call_judge_with_retry(
            provider=provider,
            prompt="test prompt",
            valid_scores={0, 1},
            judge_id="claude-haiku-4-5_groundedness",
            rubric_version="abc",
            prompt_seed=0,
            system_output_hash="def",
            item_id="item_001",
        )
        assert result.abstained
        assert result.reasoning.startswith(ABSTAIN_REASON_OUT_OF_RANGE)

    @pytest.mark.asyncio
    async def test_provider_rate_limit_abstains_with_prefix(self):
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.side_effect = ProviderRateLimitError("exhausted")

        result = await _call_judge_with_retry(
            provider=provider,
            prompt="test prompt",
            valid_scores={0, 1},
            judge_id="claude-haiku-4-5_groundedness",
            rubric_version="abc",
            prompt_seed=0,
            system_output_hash="def",
            item_id="item_001",
        )
        assert result.abstained
        assert result.reasoning.startswith(ABSTAIN_REASON_PROVIDER_EXHAUSTED)

    @pytest.mark.asyncio
    async def test_unknown_exception_reraises(self):
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.side_effect = ValueError("caller bug")

        with pytest.raises(ValueError, match="caller bug"):
            await _call_judge_with_retry(
                provider=provider,
                prompt="test prompt",
                valid_scores={0, 1},
                judge_id="x",
                rubric_version="abc",
                prompt_seed=0,
                system_output_hash="def",
                item_id="item_001",
            )

    @pytest.mark.asyncio
    async def test_genuine_unknown_score_passes_through(self):
        # Rubric allows abstain — model returns "Unknown" — no retry, no prefix
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.return_value = _mk_response(
            json.dumps(
                {
                    "reasoning": "genuinely uncertain",
                    "evidence_quotes": [],
                    "score": "Unknown",
                }
            )
        )

        result = await _call_judge_with_retry(
            provider=provider,
            prompt="test prompt",
            valid_scores={0, 1},
            judge_id="x",
            rubric_version="abc",
            prompt_seed=0,
            system_output_hash="def",
            item_id="item_001",
            abstain_allowed=True,
        )
        assert result.abstained
        assert result.reasoning == "genuinely uncertain"
        # No structured prefix on genuine abstain
        assert not result.reasoning.startswith(ABSTAIN_REASON_PROVIDER_EXHAUSTED)
        assert not result.reasoning.startswith(ABSTAIN_REASON_SCHEMA_PARSE)
        assert provider.complete.await_count == 1
