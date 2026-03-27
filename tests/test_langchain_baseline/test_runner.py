"""Tests for LangChain evaluation runner."""

from unittest.mock import AsyncMock, MagicMock

from agent_bench.langchain_baseline.runner import (
    extract_tools_used,
    run_langchain_evaluation,
)
from agent_bench.langchain_baseline.tools import LangChainSearchTool

# --- Unit tests for helper functions ---


def test_extract_tools_used_from_intermediate_steps():
    step1_action = MagicMock()
    step1_action.tool = "search_documents"
    step2_action = MagicMock()
    step2_action.tool = "calculator"

    steps = [(step1_action, "result1"), (step2_action, "result2")]
    assert extract_tools_used(steps) == ["search_documents", "calculator"]


def test_extract_tools_used_empty_steps():
    assert extract_tools_used([]) == []


# --- Integration test with mock agent executor ---


async def test_runner_produces_eval_results():
    # The runner calls reset() before each question, so pre-populating state
    # won't work. Instead, simulate what happens when the agent calls the
    # search tool during execution: the ainvoke side-effect populates metadata.
    mock_lc_retriever = MagicMock()
    search_tool = LangChainSearchTool(mock_lc_retriever)

    def _populate_search_state(*args, **kwargs):
        """Simulate the search tool populating metadata during agent execution."""
        search_tool.last_ranked_sources.append("fastapi_path_params.md")
        search_tool.last_source_chunks.append("Path params use curly braces.")
        search_tool.last_sources.append("fastapi_path_params.md")
        return {
            "output": "Path params use curly braces. [source: fastapi_path_params.md]",
            "intermediate_steps": [
                (MagicMock(tool="search_documents"), "tool output"),
            ],
        }

    agent_executor = MagicMock()
    agent_executor.ainvoke = AsyncMock(side_effect=_populate_search_state)

    golden_path = "agent_bench/evaluation/datasets/tech_docs_golden.json"

    results = await run_langchain_evaluation(
        agent_executor=agent_executor,
        search_tool_state=search_tool,
        golden_path=golden_path,
        provider_name="openai",
        max_questions=2,  # only run first 2 for speed
    )

    assert len(results) == 2
    r = results[0]
    assert r.question_id == "q001"
    assert r.question == "How do you define a path parameter in FastAPI?"
    assert r.category == "retrieval"
    assert r.answer != ""
    # Verify metadata actually propagated (not zeroed by reset)
    assert r.retrieval_precision > 0.0
    assert r.retrieval_recall > 0.0
    assert r.retrieved_sources == ["fastapi_path_params.md"]


async def test_runner_handles_agent_error():
    agent_executor = MagicMock()
    agent_executor.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))

    mock_lc_retriever = MagicMock()
    search_tool = LangChainSearchTool(mock_lc_retriever)

    golden_path = "agent_bench/evaluation/datasets/tech_docs_golden.json"

    results = await run_langchain_evaluation(
        agent_executor=agent_executor,
        search_tool_state=search_tool,
        golden_path=golden_path,
        provider_name="openai",
        max_questions=1,
    )

    assert len(results) == 1
    assert "ERROR" in results[0].answer
    assert results[0].tool_calls_made == 0
