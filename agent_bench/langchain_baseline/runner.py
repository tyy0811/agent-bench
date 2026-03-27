"""Evaluation runner: LangChain agent -> EvalResult (same format as existing harness)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.callbacks.usage import UsageMetadataCallbackHandler

from agent_bench.core.types import TokenUsage
from agent_bench.evaluation.harness import EvalResult, load_golden_dataset
from agent_bench.evaluation.metrics import (
    citation_accuracy,
    grounded_refusal,
    keyword_hit_rate,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
)

if TYPE_CHECKING:
    from langchain.agents import AgentExecutor

    from agent_bench.langchain_baseline.tools import LangChainSearchTool


def extract_tools_used(intermediate_steps: list) -> list[str]:
    """Extract tool names from LangChain intermediate steps.

    Each step is a (AgentAction, observation) tuple.
    """
    return [step[0].tool for step in intermediate_steps if hasattr(step[0], "tool")]


def _estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_mtok: float,
    output_cost_per_mtok: float,
) -> float:
    return (input_tokens * input_cost_per_mtok + output_tokens * output_cost_per_mtok) / 1_000_000


async def run_langchain_evaluation(
    agent_executor: AgentExecutor,
    search_tool_state: LangChainSearchTool,
    golden_path: str | Path,
    provider_name: str,
    max_questions: int | None = None,
    input_cost_per_mtok: float = 0.0,
    output_cost_per_mtok: float = 0.0,
) -> list[EvalResult]:
    """Run golden dataset through LangChain agent, producing EvalResult objects.

    Uses the same metric functions as agent_bench.evaluation.harness, so results
    are directly comparable and can be fed into generate_report().

    Args:
        agent_executor: Configured LangChain AgentExecutor.
        search_tool_state: The LangChainSearchTool instance (for metadata capture).
        golden_path: Path to the golden dataset JSON.
        provider_name: Provider name for reporting (e.g. "openai").
        max_questions: Limit number of questions (for testing). None = all.
        input_cost_per_mtok: Input token cost per million tokens (from config).
        output_cost_per_mtok: Output token cost per million tokens (from config).
    """
    questions = load_golden_dataset(golden_path)
    if max_questions is not None:
        questions = questions[:max_questions]

    results: list[EvalResult] = []

    for q in questions:
        search_tool_state.reset()
        # Fresh handler per question — reads AIMessage.usage_metadata,
        # which is populated by both OpenAI and Anthropic adapters.
        token_tracker = UsageMetadataCallbackHandler()
        start = time.perf_counter()

        try:
            response = await agent_executor.ainvoke(
                {"input": q.question},
                config={"callbacks": [token_tracker]},
            )
            latency_ms = (time.perf_counter() - start) * 1000

            answer = response.get("output", "")
            steps = response.get("intermediate_steps", [])
            tools_used = extract_tools_used(steps)

            ranked_sources = list(search_tool_state.last_ranked_sources)
            deduped_sources = list(search_tool_state.last_sources)

            # usage_metadata is keyed by model name, e.g.
            # {"gpt-4o-mini-2024-07-18": {"input_tokens": 8, ...}}
            # Sum across all models (typically one) to get totals.
            input_toks = sum(
                v.get("input_tokens", 0)
                for v in token_tracker.usage_metadata.values()
            )
            output_toks = sum(
                v.get("output_tokens", 0)
                for v in token_tracker.usage_metadata.values()
            )

            result = EvalResult(
                question_id=q.id,
                question=q.question,
                category=q.category,
                difficulty=q.difficulty,
                retrieval_precision=retrieval_precision_at_k(
                    ranked_sources, q.expected_sources
                ),
                retrieval_recall=retrieval_recall_at_k(
                    ranked_sources, q.expected_sources
                ),
                keyword_hit_rate=keyword_hit_rate(answer, q.expected_answer_keywords),
                has_source_citation=len(deduped_sources) > 0,
                grounded_refusal=grounded_refusal(
                    answer, q.category, deduped_sources
                ),
                citation_accuracy=citation_accuracy(answer, deduped_sources),
                calculator_used_correctly=(
                    ("calculator" in tools_used) if q.requires_calculator else True
                ),
                tool_calls_made=len(tools_used),
                latency_ms=latency_ms,
                tokens_used=TokenUsage(
                    input_tokens=input_toks,
                    output_tokens=output_toks,
                    estimated_cost_usd=_estimate_cost(
                        input_toks,
                        output_toks,
                        input_cost_per_mtok,
                        output_cost_per_mtok,
                    ),
                ),
                answer=answer,
                retrieved_sources=ranked_sources,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            result = EvalResult(
                question_id=q.id,
                question=q.question,
                category=q.category,
                difficulty=q.difficulty,
                retrieval_precision=0.0,
                retrieval_recall=0.0,
                keyword_hit_rate=0.0,
                has_source_citation=False,
                grounded_refusal=q.category != "out_of_scope",
                citation_accuracy=1.0,
                calculator_used_correctly=not q.requires_calculator,
                tool_calls_made=0,
                latency_ms=latency_ms,
                tokens_used=TokenUsage(
                    input_tokens=0, output_tokens=0, estimated_cost_usd=0.0
                ),
                answer=f"ERROR: {e}",
                retrieved_sources=[],
            )

        results.append(result)

    return results
