"""Tests for the harness migration to the new judge layer."""

from __future__ import annotations

from agent_bench.core.config import EvaluationConfig


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
