"""Tests for the agent orchestrator."""

from __future__ import annotations

import pytest

from agent_bench.agents.orchestrator import AgentResponse, Orchestrator
from agent_bench.core.provider import MockProvider
from agent_bench.core.types import (
    CompletionResponse,
    Role,
    TokenUsage,
    ToolCall,
)
from agent_bench.tools.base import Tool, ToolOutput
from agent_bench.tools.calculator import CalculatorTool
from agent_bench.tools.registry import ToolRegistry

# --- Helpers ---


class FakeSearchTool(Tool):
    """Deterministic search tool for orchestrator tests."""

    name = "search_documents"
    description = "Search docs"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    async def execute(self, **kwargs: object) -> ToolOutput:
        return ToolOutput(
            success=True,
            result="[1] (fastapi_path_params.md): Path parameters use curly braces.",
            metadata={"sources": ["fastapi_path_params.md"]},
        )


class AlwaysToolCallProvider(MockProvider):
    """Provider that always returns tool_calls, never a final answer."""

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=1024):
        self.call_count += 1
        if tools is None:
            # Forced final call (no tools) — return text
            return CompletionResponse(
                content="Forced answer after max iterations.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=100, output_tokens=20, estimated_cost_usd=0.0001),
                provider="mock",
                model="mock-1",
                latency_ms=1.0,
            )
        return CompletionResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id=f"call_{self.call_count}",
                    name="search_documents",
                    arguments={"query": "test"},
                )
            ],
            usage=TokenUsage(input_tokens=100, output_tokens=20, estimated_cost_usd=0.0001),
            provider="mock",
            model="mock-1",
            latency_ms=1.0,
        )


class MultiSearchProvider(MockProvider):
    """Provider that searches twice (different queries), then answers."""

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=1024):
        self.call_count += 1
        tool_results = [m for m in messages if m.role == Role.TOOL]

        if tools and len(tool_results) == 0:
            return CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(id="call_1", name="search_documents", arguments={"query": "first"})
                ],
                usage=TokenUsage(input_tokens=100, output_tokens=20, estimated_cost_usd=0.0001),
                provider="mock",
                model="mock-1",
                latency_ms=1.0,
            )
        elif tools and len(tool_results) == 1:
            return CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(id="call_2", name="search_documents", arguments={"query": "second"})
                ],
                usage=TokenUsage(input_tokens=150, output_tokens=25, estimated_cost_usd=0.0002),
                provider="mock",
                model="mock-1",
                latency_ms=1.0,
            )
        else:
            return CompletionResponse(
                content="Answer from two searches. [source: fastapi_path_params.md]",
                tool_calls=[],
                usage=TokenUsage(input_tokens=200, output_tokens=50, estimated_cost_usd=0.0003),
                provider="mock",
                model="mock-1",
                latency_ms=2.0,
            )


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(FakeSearchTool())
    registry.register(CalculatorTool())
    return registry


SYSTEM_PROMPT = "You are a helpful assistant."


# --- Tests ---


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_produces_agent_response_with_all_fields(self):
        """Orchestrator returns AgentResponse with all required fields."""
        orchestrator = Orchestrator(
            provider=MockProvider(), registry=_make_registry(), max_iterations=3
        )
        response = await orchestrator.run("How do path params work?", SYSTEM_PROMPT)

        assert isinstance(response, AgentResponse)
        assert len(response.answer) > 0
        assert response.iterations >= 1
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.latency_ms > 0
        assert isinstance(response.sources, list)
        assert isinstance(response.tools_used, list)

    @pytest.mark.asyncio
    async def test_respects_max_iterations(self):
        """When provider always returns tool_calls, orchestrator stops at max_iterations."""
        provider = AlwaysToolCallProvider()
        orchestrator = Orchestrator(provider=provider, registry=_make_registry(), max_iterations=2)
        response = await orchestrator.run("test question", SYSTEM_PROMPT)

        # 2 iterations of tool calls + 1 forced final call = 3 provider calls total
        assert provider.call_count == 3
        assert response.iterations == 2
        assert response.answer == "Forced answer after max iterations."

    @pytest.mark.asyncio
    async def test_accumulates_sources_from_multiple_searches(self):
        """Sources from multiple search calls are accumulated and deduplicated."""
        orchestrator = Orchestrator(
            provider=MultiSearchProvider(), registry=_make_registry(), max_iterations=3
        )
        response = await orchestrator.run("multi search question", SYSTEM_PROMPT)

        # FakeSearchTool always returns fastapi_path_params.md
        assert len(response.sources) == 1  # deduplicated
        assert response.sources[0].source == "fastapi_path_params.md"
        assert response.tools_used.count("search_documents") == 2
        # Token usage accumulated across 3 provider calls
        assert response.usage.input_tokens == 100 + 150 + 200
        assert response.usage.output_tokens == 20 + 25 + 50

    @pytest.mark.asyncio
    async def test_deterministic_output(self):
        """Fixed question + MockProvider → exact expected answer."""
        orchestrator = Orchestrator(
            provider=MockProvider(), registry=_make_registry(), max_iterations=3
        )
        response = await orchestrator.run("How do path params work?", SYSTEM_PROMPT)

        # MockProvider: first call returns tool_calls for search_documents,
        # second call (with tool results) returns the canned answer
        assert "path parameters" in response.answer.lower()
        assert "[source: fastapi_path_params.md]" in response.answer
        assert "search_documents" in response.tools_used
        assert response.iterations == 2


class TestOrchestratorIntegration:
    """Integration test using real SearchTool + Retriever + HybridStore."""

    @pytest.mark.asyncio
    async def test_real_rag_path(self, test_retriever):
        """Orchestrator with MockProvider + real SearchTool/Retriever returns RAG results."""
        from agent_bench.tools.search import SearchTool

        registry = ToolRegistry()
        registry.register(SearchTool(retriever=test_retriever))
        registry.register(CalculatorTool())

        orchestrator = Orchestrator(provider=MockProvider(), registry=registry, max_iterations=3)
        response = await orchestrator.run(
            "How do path params work?", SYSTEM_PROMPT, top_k=3, strategy="hybrid"
        )

        # MockProvider drives the loop, but the real SearchTool executes
        # against the real Retriever/HybridStore and returns real chunks
        assert isinstance(response, AgentResponse)
        assert len(response.answer) > 0
        assert "search_documents" in response.tools_used
        # Sources come from the real store (sample_chunks in conftest)
        assert len(response.sources) > 0
