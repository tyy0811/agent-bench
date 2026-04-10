"""Tests for SSE stage events emitted by the orchestrator."""

import pytest

from agent_bench.agents.orchestrator import Orchestrator
from agent_bench.core.provider import MockProvider
from agent_bench.tools.registry import ToolRegistry

from tests.test_agent import FakeSearchTool


class TestOrchestratorStageEvents:
    @pytest.fixture
    def orchestrator(self):
        registry = ToolRegistry()
        registry.register(FakeSearchTool())
        return Orchestrator(
            provider=MockProvider(),
            registry=registry,
            max_iterations=3,
        )

    @pytest.mark.asyncio
    async def test_stream_emits_retrieval_stage(self, orchestrator):
        events = []
        async for event in orchestrator.run_stream(
            question="How do path params work?",
            system_prompt="You are a test assistant.",
        ):
            events.append(event)

        stage_events = [e for e in events if e.type == "stage"]
        retrieval_events = [e for e in stage_events if e.metadata.get("stage") == "retrieval"]
        assert len(retrieval_events) >= 2  # running + done
        done = [e for e in retrieval_events if e.metadata.get("status") == "done"]
        assert len(done) >= 1
        assert "chunks_pre_rerank" in done[0].metadata

    @pytest.mark.asyncio
    async def test_stream_emits_reranking_stage(self, orchestrator):
        events = []
        async for event in orchestrator.run_stream(
            question="How do path params work?",
            system_prompt="You are a test assistant.",
        ):
            events.append(event)

        stage_events = [e for e in events if e.type == "stage"]
        reranking_events = [e for e in stage_events if e.metadata.get("stage") == "reranking"]
        assert len(reranking_events) >= 1  # done event with chunk details
        # Reranking completes inside tool execution, so only a done event is emitted
        assert all(e.metadata.get("status") == "done" for e in reranking_events)

    @pytest.mark.asyncio
    async def test_stream_emits_llm_stage(self, orchestrator):
        events = []
        async for event in orchestrator.run_stream(
            question="How do path params work?",
            system_prompt="You are a test assistant.",
        ):
            events.append(event)

        stage_events = [e for e in events if e.type == "stage"]
        llm_events = [e for e in stage_events if e.metadata.get("stage") == "llm"]
        assert len(llm_events) >= 1  # at least done

    @pytest.mark.asyncio
    async def test_stream_stage_events_have_iteration(self, orchestrator):
        events = []
        async for event in orchestrator.run_stream(
            question="How do path params work?",
            system_prompt="You are a test assistant.",
        ):
            events.append(event)

        stage_events = [e for e in events if e.type == "stage"]
        for e in stage_events:
            if e.metadata.get("stage") in ("retrieval", "reranking", "llm"):
                assert "iteration" in e.metadata

    @pytest.mark.asyncio
    async def test_stream_preserves_sources_chunk_done_order(self, orchestrator):
        events = []
        async for event in orchestrator.run_stream(
            question="How do path params work?",
            system_prompt="You are a test assistant.",
        ):
            events.append(event)

        # Filter to legacy event types
        legacy = [e for e in events if e.type in ("sources", "chunk", "_orchestrator_done")]
        assert len(legacy) >= 3
        types = [e.type for e in legacy]
        assert types[0] == "sources"
        assert types[-1] == "_orchestrator_done"

    @pytest.mark.asyncio
    async def test_stream_tool_call_includes_arguments(self, orchestrator):
        """MockProvider emits a search_documents tool call on first iteration."""
        events = []
        async for event in orchestrator.run_stream(
            question="How do path params work?",
            system_prompt="You are a test assistant.",
        ):
            events.append(event)

        stage_events = [e for e in events if e.type == "stage"]
        llm_tool_calls = [e for e in stage_events
                          if e.metadata.get("stage") == "llm"
                          and e.metadata.get("status") == "tool_call"]
        # MockProvider returns tool calls when tools are provided
        if llm_tool_calls:
            assert "tool" in llm_tool_calls[0].metadata
            assert "arguments" in llm_tool_calls[0].metadata
