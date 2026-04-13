"""Automated evaluation runner over the golden dataset."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from agent_bench.agents.orchestrator import Orchestrator
from agent_bench.core.provider import LLMProvider
from agent_bench.core.types import TokenUsage
from agent_bench.evaluation.metrics import (
    calculator_used_when_expected,
    citation_accuracy,
    grounded_refusal,
    keyword_hit_rate,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
    source_presence,
    tool_call_count,
)


class GoldenQuestion(BaseModel):
    id: str
    question: str
    expected_answer_keywords: list[str]
    expected_sources: list[str]
    category: str
    difficulty: str
    requires_calculator: bool
    reference_answer: str = ""
    # Multi-corpus schema v2 (optional)
    source_chunk_ids: list[str] = []
    source_snippets: list[str] = []
    question_type: str = ""
    is_multi_hop: bool = False
    # Authoring-time anchors for pre-ingestion golden datasets; index-aligned
    # with source_snippets. source_sections[i] == "" means the snippet lives in
    # page lede content above the first H2/H3 — this is allowed, not a missing
    # value. Backfill matches on source_snippets, not on these fields.
    source_pages: list[str] = Field(default_factory=list)
    source_sections: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    question_id: str
    question: str
    category: str
    difficulty: str
    # Deterministic
    retrieval_precision: float
    retrieval_recall: float
    keyword_hit_rate: float
    has_source_citation: bool
    grounded_refusal: bool
    citation_accuracy: float
    calculator_used_correctly: bool
    tool_calls_made: int
    latency_ms: float
    tokens_used: TokenUsage
    # Raw answer for reporting
    answer: str = ""
    retrieved_sources: list[str] = []
    # LLM judge (None if not run)
    faithfulness: float | None = None
    correctness: float | None = None


def load_golden_dataset(path: str | Path) -> list[GoldenQuestion]:
    """Load golden questions from JSON.

    Supports two formats:
    - Legacy flat list: [{...}, {...}]
    - Nested with header: {"corpus": ..., "version": ..., "questions": [...]}
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "questions" in data:
        items = data["questions"]
    else:
        raise ValueError(
            f"Unrecognized golden dataset format at {path}: "
            "expected list or dict with 'questions' key",
        )
    return [GoldenQuestion.model_validate(q) for q in items]


async def run_evaluation(
    orchestrator: Orchestrator,
    system_prompt: str,
    golden_path: str | Path,
    judge_provider: LLMProvider | None = None,
) -> list[EvalResult]:
    """Run the evaluation harness over the golden dataset.

    Args:
        orchestrator: Configured orchestrator with provider + tools.
        system_prompt: System prompt for the task.
        golden_path: Path to the golden dataset JSON.
        judge_provider: Optional LLM provider for faithfulness/correctness scoring.

    Returns:
        List of EvalResult, one per question.
    """
    questions = load_golden_dataset(golden_path)
    results: list[EvalResult] = []

    for q in questions:
        # Run the agent
        agent_response = await orchestrator.run(
            question=q.question,
            system_prompt=system_prompt,
        )

        # Use ranked_sources for retrieval metrics (preserves rank order)
        ranked_sources = agent_response.ranked_sources
        deduped_sources = [s.source for s in agent_response.sources]

        # Compute deterministic metrics
        result = EvalResult(
            question_id=q.id,
            question=q.question,
            category=q.category,
            difficulty=q.difficulty,
            retrieval_precision=retrieval_precision_at_k(ranked_sources, q.expected_sources),
            retrieval_recall=retrieval_recall_at_k(ranked_sources, q.expected_sources),
            keyword_hit_rate=keyword_hit_rate(agent_response.answer, q.expected_answer_keywords),
            has_source_citation=source_presence(agent_response),
            grounded_refusal=grounded_refusal(agent_response.answer, q.category, deduped_sources),
            citation_accuracy=citation_accuracy(agent_response.answer, deduped_sources),
            calculator_used_correctly=calculator_used_when_expected(
                agent_response, q.requires_calculator
            ),
            tool_calls_made=tool_call_count(agent_response),
            latency_ms=agent_response.latency_ms,
            tokens_used=agent_response.usage,
            answer=agent_response.answer,
            retrieved_sources=ranked_sources,
        )

        # Optional LLM judge
        if judge_provider is not None and q.category != "out_of_scope":
            from agent_bench.evaluation.metrics import answer_correctness, answer_faithfulness

            result.faithfulness = await answer_faithfulness(
                answer=agent_response.answer,
                source_chunks=agent_response.source_chunks,
                judge_provider=judge_provider,
            )
            if q.reference_answer:
                result.correctness = await answer_correctness(
                    answer=agent_response.answer,
                    reference_answer=q.reference_answer,
                    judge_provider=judge_provider,
                )

        results.append(result)

    return results
