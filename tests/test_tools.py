"""Tests for tool system: registry, calculator, search, and schema generation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_bench.tools.calculator import CalculatorTool
from agent_bench.tools.registry import ToolRegistry
from agent_bench.tools.search import SearchTool

# --- Mock retriever for SearchTool tests ---


@dataclass
class MockChunk:
    content: str
    source: str


@dataclass
class MockSearchResult:
    chunk: MockChunk
    score: float


class MockRetrievalResult:
    """Mimics RetrievalResult for test mocks."""
    def __init__(self, results: list, pre_rerank_count: int = 0) -> None:
        self.results = results
        self.pre_rerank_count = pre_rerank_count


class MockRetriever:
    """Fake retriever that returns canned results."""

    def __init__(self, results: list[MockSearchResult] | None = None) -> None:
        self._results = results or []

    async def search(
        self, query: str, top_k: int = 5, strategy: str | None = None
    ) -> MockRetrievalResult:
        return MockRetrievalResult(results=self._results[:top_k])


# --- Registry tests ---


class TestToolRegistry:
    def test_register_and_retrieve(self):
        registry = ToolRegistry()
        tool = CalculatorTool()
        registry.register(tool)
        assert registry.get("calculator") is tool

    def test_get_unknown_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = await registry.execute("nonexistent", query="test")
        assert result.success is False
        assert "Unknown tool: nonexistent" in result.result

    @pytest.mark.asyncio
    async def test_execute_registered_tool(self):
        """Registry dispatches to a registered tool and returns its output."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        result = await registry.execute("calculator", expression="1 + 2")
        assert result.success is True
        assert result.result == "3"

    def test_get_definitions(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(SearchTool(retriever=MockRetriever()))
        defs = registry.get_definitions()
        assert len(defs) == 2
        names = {d.name for d in defs}
        assert names == {"calculator", "search_documents"}


# --- Calculator tests ---


class TestCalculatorTool:
    @pytest.mark.asyncio
    async def test_valid_expression(self):
        calc = CalculatorTool()
        result = await calc.execute(expression="2 + 3 * 4")
        assert result.success is True
        assert result.result == "14"

    @pytest.mark.asyncio
    async def test_float_expression(self):
        calc = CalculatorTool()
        result = await calc.execute(expression="10 / 3")
        assert result.success is True
        assert float(result.result) == pytest.approx(3.333333, rel=1e-4)

    @pytest.mark.asyncio
    async def test_rejects_import(self):
        calc = CalculatorTool()
        result = await calc.execute(expression="__import__('os').system('ls')")
        assert result.success is False
        assert "Could not evaluate" in result.result

    @pytest.mark.asyncio
    async def test_rejects_exec(self):
        calc = CalculatorTool()
        result = await calc.execute(expression="exec('print(1)')")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_empty_expression(self):
        calc = CalculatorTool()
        result = await calc.execute(expression="")
        assert result.success is False

    def test_definition_produces_valid_schema(self):
        calc = CalculatorTool()
        defn = calc.definition()
        assert defn.name == "calculator"
        assert defn.parameters["type"] == "object"
        assert "expression" in defn.parameters["properties"]
        assert "expression" in defn.parameters["required"]


# --- Search tool tests ---


class TestSearchTool:
    @pytest.mark.asyncio
    async def test_returns_formatted_results(self):
        retriever = MockRetriever(
            results=[
                MockSearchResult(
                    chunk=MockChunk(
                        content="Path parameters are defined using curly braces.",
                        source="fastapi_path_params.md",
                    ),
                    score=0.95,
                ),
                MockSearchResult(
                    chunk=MockChunk(
                        content="Query parameters are automatically parsed.",
                        source="fastapi_query_params.md",
                    ),
                    score=0.82,
                ),
            ]
        )
        tool = SearchTool(retriever=retriever)
        result = await tool.execute(query="path parameters")

        assert result.success is True
        assert "[1] (fastapi_path_params.md):" in result.result
        assert "[2] (fastapi_query_params.md):" in result.result
        assert result.metadata["sources"] == [
            "fastapi_path_params.md",
            "fastapi_query_params.md",
        ]

    @pytest.mark.asyncio
    async def test_empty_results(self):
        tool = SearchTool(retriever=MockRetriever(results=[]))
        result = await tool.execute(query="nonexistent topic")
        assert result.success is True
        assert "No relevant documents found" in result.result
        assert result.metadata["sources"] == []

    @pytest.mark.asyncio
    async def test_deduplicates_sources(self):
        retriever = MockRetriever(
            results=[
                MockSearchResult(
                    chunk=MockChunk(content="Chunk 1", source="same_file.md"),
                    score=0.9,
                ),
                MockSearchResult(
                    chunk=MockChunk(content="Chunk 2", source="same_file.md"),
                    score=0.8,
                ),
            ]
        )
        tool = SearchTool(retriever=retriever)
        result = await tool.execute(query="test")
        assert result.metadata["sources"] == ["same_file.md"]

    @pytest.mark.asyncio
    async def test_malformed_top_k_falls_back_to_default(self):
        """Model-generated bad top_k (e.g. 'five') should not crash."""
        retriever = MockRetriever(
            results=[
                MockSearchResult(
                    chunk=MockChunk(content="Some content", source="test.md"),
                    score=0.9,
                ),
            ]
        )
        tool = SearchTool(retriever=retriever)
        result = await tool.execute(query="test", top_k="five")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_empty_query(self):
        tool = SearchTool(retriever=MockRetriever())
        result = await tool.execute(query="")
        assert result.success is False

    def test_definition_produces_valid_schema(self):
        tool = SearchTool(retriever=MockRetriever())
        defn = tool.definition()
        assert defn.name == "search_documents"
        assert defn.parameters["type"] == "object"
        assert "query" in defn.parameters["properties"]
        assert "query" in defn.parameters["required"]


# --- Refusal gate tests ---


class TestRefusalGate:
    """Tests for grounded refusal based on retrieval score threshold."""

    def _make_results(self, scores: list[float]) -> list[MockSearchResult]:
        return [
            MockSearchResult(
                chunk=MockChunk(content=f"Content {i}", source=f"doc_{i}.md"),
                score=s,
            )
            for i, s in enumerate(scores)
        ]

    @pytest.mark.asyncio
    async def test_refusal_out_of_scope(self):
        """Low-scoring results below threshold trigger refusal."""
        retriever = MockRetriever(results=self._make_results([0.005, 0.003]))
        tool = SearchTool(retriever=retriever, refusal_threshold=0.02)
        result = await tool.execute(query="how to cook pasta")

        assert result.success is True
        assert "No relevant documents found" in result.result
        assert result.metadata["refused"] is True
        assert result.metadata["sources"] == []

    @pytest.mark.asyncio
    async def test_no_refusal_in_scope(self):
        """High-scoring results above threshold proceed normally."""
        retriever = MockRetriever(results=self._make_results([0.03, 0.025]))
        tool = SearchTool(retriever=retriever, refusal_threshold=0.02)
        result = await tool.execute(query="FastAPI authentication")

        assert result.success is True
        assert "No relevant documents found" not in result.result
        assert result.metadata.get("refused") is None
        assert len(result.metadata["sources"]) == 2

    @pytest.mark.asyncio
    async def test_refusal_metadata(self):
        """Refused response includes max_score and refused=True."""
        retriever = MockRetriever(results=self._make_results([0.01, 0.008]))
        tool = SearchTool(retriever=retriever, refusal_threshold=0.02)
        result = await tool.execute(query="unrelated topic")

        assert result.metadata["max_score"] == 0.01
        assert result.metadata["refused"] is True

    @pytest.mark.asyncio
    async def test_threshold_zero_disables(self):
        """threshold=0.0 (default) never refuses, preserving V1 behavior."""
        retriever = MockRetriever(results=self._make_results([0.001]))
        tool = SearchTool(retriever=retriever, refusal_threshold=0.0)
        result = await tool.execute(query="anything")

        assert result.success is True
        assert "No relevant documents found" not in result.result
        assert result.metadata.get("refused") is None

    @pytest.mark.asyncio
    async def test_threshold_configurable(self):
        """Different threshold values change refusal behavior."""
        results = self._make_results([0.015, 0.012])
        retriever = MockRetriever(results=results)

        # With threshold=0.01 -> passes (max_score 0.015 > 0.01)
        tool_low = SearchTool(retriever=retriever, refusal_threshold=0.01)
        result_low = await tool_low.execute(query="test")
        assert result_low.metadata.get("refused") is None

        # With threshold=0.02 -> refuses (max_score 0.015 < 0.02)
        tool_high = SearchTool(retriever=retriever, refusal_threshold=0.02)
        result_high = await tool_high.execute(query="test")
        assert result_high.metadata["refused"] is True

    @pytest.mark.asyncio
    async def test_max_score_in_normal_metadata(self):
        """Non-refused responses also include max_score in metadata."""
        retriever = MockRetriever(results=self._make_results([0.05, 0.03]))
        tool = SearchTool(retriever=retriever, refusal_threshold=0.02)
        result = await tool.execute(query="test")

        assert result.metadata["max_score"] == 0.05
