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
