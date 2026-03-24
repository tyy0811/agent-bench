"""Tests for evaluation metrics, harness, and report generation."""

from __future__ import annotations

import pytest

from agent_bench.agents.orchestrator import AgentResponse, SourceReference
from agent_bench.core.types import TokenUsage
from agent_bench.evaluation.harness import EvalResult, load_golden_dataset
from agent_bench.evaluation.metrics import (
    calculator_used_when_expected,
    citation_accuracy,
    grounded_refusal,
    keyword_hit_rate,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
    source_presence,
)
from agent_bench.evaluation.report import generate_report

# --- Metrics tests ---


class TestRetrievalMetrics:
    def test_precision_at_k_perfect(self):
        assert retrieval_precision_at_k(["a.md", "b.md"], ["a.md", "b.md"]) == 1.0

    def test_precision_at_k_partial(self):
        assert retrieval_precision_at_k(["a.md", "b.md", "c.md"], ["a.md"]) == pytest.approx(1 / 3)

    def test_precision_at_k_empty_retrieved(self):
        assert retrieval_precision_at_k([], ["a.md"]) == 0.0

    def test_recall_at_k_perfect(self):
        assert retrieval_recall_at_k(["a.md", "b.md", "c.md"], ["a.md", "b.md"]) == 1.0

    def test_recall_at_k_partial(self):
        assert retrieval_recall_at_k(["a.md"], ["a.md", "b.md"]) == 0.5

    def test_recall_at_k_empty_expected(self):
        assert retrieval_recall_at_k(["a.md"], []) == 0.0

    def test_precision_uses_ranked_sources_with_duplicates(self):
        """Ranked sources may have duplicates — precision should count correctly."""
        retrieved = ["a.md", "a.md", "b.md", "c.md", "d.md"]
        expected = ["a.md"]
        # 2 out of 5 retrieved are "a.md"
        assert retrieval_precision_at_k(retrieved, expected, k=5) == pytest.approx(2 / 5)


class TestKeywordMetrics:
    def test_keyword_hit_rate_all_match(self):
        assert keyword_hit_rate("curly braces in path", ["curly braces", "path"]) == 1.0

    def test_keyword_hit_rate_none_match(self):
        assert keyword_hit_rate("something else", ["curly", "braces"]) == 0.0

    def test_keyword_hit_rate_case_insensitive(self):
        assert keyword_hit_rate("CORSMiddleware", ["corsmiddleware"]) == 1.0


class TestSourcePresence:
    def test_has_sources(self):
        resp = AgentResponse(
            answer="test",
            sources=[SourceReference(source="a.md")],
            iterations=1,
            usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
            latency_ms=1.0,
        )
        assert source_presence(resp) is True

    def test_no_sources(self):
        resp = AgentResponse(
            answer="test",
            sources=[],
            iterations=1,
            usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
            latency_ms=1.0,
        )
        assert source_presence(resp) is False


class TestGroundedRefusal:
    def test_out_of_scope_with_refusal(self):
        assert (
            grounded_refusal("The documentation does not contain this info.", "out_of_scope", [])
            is True
        )

    def test_out_of_scope_without_refusal(self):
        assert grounded_refusal("Here is how you do it...", "out_of_scope", []) is False

    def test_in_scope_always_true(self):
        assert grounded_refusal("any answer", "retrieval", ["a.md"]) is True


class TestCitationAccuracy:
    def test_all_citations_valid(self):
        answer = "Info from [source: a.md] and [source: b.md]."
        assert citation_accuracy(answer, ["a.md", "b.md"]) == 1.0

    def test_hallucinated_citation(self):
        answer = "Info from [source: fake.md]."
        assert citation_accuracy(answer, ["a.md"]) == 0.0

    def test_no_citations(self):
        assert citation_accuracy("No citations here.", ["a.md"]) == 1.0


class TestCalculatorMetric:
    def test_calculator_used_when_required(self):
        resp = AgentResponse(
            answer="9",
            tools_used=["search_documents", "calculator"],
            iterations=2,
            usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
            latency_ms=1.0,
        )
        assert calculator_used_when_expected(resp, requires_calculator=True) is True

    def test_calculator_not_used_when_required(self):
        resp = AgentResponse(
            answer="9",
            tools_used=["search_documents"],
            iterations=1,
            usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
            latency_ms=1.0,
        )
        assert calculator_used_when_expected(resp, requires_calculator=True) is False

    def test_not_required_always_true(self):
        resp = AgentResponse(
            answer="test",
            tools_used=[],
            iterations=1,
            usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
            latency_ms=1.0,
        )
        assert calculator_used_when_expected(resp, requires_calculator=False) is True


# --- Golden dataset loading ---


class TestGoldenDataset:
    def test_load_golden_dataset(self):
        questions = load_golden_dataset("agent_bench/evaluation/datasets/tech_docs_golden.json")
        assert len(questions) == 27
        # Check distribution
        categories = [q.category for q in questions]
        assert categories.count("out_of_scope") == 5
        assert categories.count("calculation") == 3
        # All have required fields
        for q in questions:
            assert q.id
            assert q.question
            assert q.expected_answer_keywords


# --- Report generation ---


class TestReportGeneration:
    def _make_results(self) -> list[EvalResult]:
        usage = TokenUsage(input_tokens=100, output_tokens=50, estimated_cost_usd=0.001)
        return [
            EvalResult(
                question_id="q001",
                question="Test question?",
                category="retrieval",
                difficulty="easy",
                retrieval_precision=0.8,
                retrieval_recall=1.0,
                keyword_hit_rate=0.75,
                has_source_citation=True,
                grounded_refusal=True,
                citation_accuracy=1.0,
                calculator_used_correctly=True,
                tool_calls_made=2,
                latency_ms=100.0,
                tokens_used=usage,
                answer="Test answer",
                retrieved_sources=["a.md"],
            ),
            EvalResult(
                question_id="q002",
                question="Out of scope?",
                category="out_of_scope",
                difficulty="easy",
                retrieval_precision=0.0,
                retrieval_recall=0.0,
                keyword_hit_rate=0.5,
                has_source_citation=False,
                grounded_refusal=True,
                citation_accuracy=1.0,
                calculator_used_correctly=True,
                tool_calls_made=1,
                latency_ms=50.0,
                tokens_used=usage,
                answer="Does not contain",
                retrieved_sources=[],
            ),
        ]

    def test_report_contains_required_sections(self):
        report = generate_report(self._make_results(), provider_name="test")
        assert "## Aggregate Metrics" in report
        assert "## By Category" in report
        assert "## By Difficulty" in report
        assert "## Chunking Strategy Comparison" in report
        assert "## Failure Analysis" in report
        assert "## Per-Question Results" in report

    def test_report_contains_metrics(self):
        report = generate_report(self._make_results(), provider_name="test")
        assert "Retrieval P@5" in report
        assert "Grounded Refusal Rate" in report
        assert "Citation Accuracy" in report
