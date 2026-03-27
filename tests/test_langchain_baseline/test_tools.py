"""Tests for LangChain tool wrappers."""

from unittest.mock import AsyncMock, MagicMock

from langchain_core.documents import Document as LCDocument

from agent_bench.langchain_baseline.tools import LangChainSearchTool, create_calculator_tool


# --- Search tool ---


def _make_mock_lc_retriever(docs=None):
    """Mock an AgentBenchRetriever (LangChain retriever)."""
    ret = MagicMock()
    if docs is None:
        docs = [
            LCDocument(
                page_content="Path params use curly braces.",
                metadata={"source": "fastapi_path_params.md", "chunk_id": "c1", "score": 0.9},
            ),
            LCDocument(
                page_content="Query params are parsed from URL.",
                metadata={"source": "fastapi_query_params.md", "chunk_id": "c2", "score": 0.7},
            ),
        ]
    ret.ainvoke = AsyncMock(return_value=docs)
    return ret


async def test_search_tool_returns_formatted_passages():
    mock_ret = _make_mock_lc_retriever()
    search = LangChainSearchTool(mock_ret)
    tool = search.as_tool()

    result = await tool.ainvoke({"query": "path parameters"})

    assert "[1] (fastapi_path_params.md):" in result
    assert "[2] (fastapi_query_params.md):" in result
    assert "curly braces" in result


async def test_search_tool_captures_ranked_sources():
    mock_ret = _make_mock_lc_retriever()
    search = LangChainSearchTool(mock_ret)
    tool = search.as_tool()

    await tool.ainvoke({"query": "test"})

    assert search.last_ranked_sources == [
        "fastapi_path_params.md",
        "fastapi_query_params.md",
    ]


async def test_search_tool_captures_source_chunks():
    mock_ret = _make_mock_lc_retriever()
    search = LangChainSearchTool(mock_ret)
    tool = search.as_tool()

    await tool.ainvoke({"query": "test"})

    assert search.last_source_chunks == [
        "Path params use curly braces.",
        "Query params are parsed from URL.",
    ]


async def test_search_tool_deduplicates_sources():
    docs = [
        LCDocument(page_content="A", metadata={"source": "x.md", "chunk_id": "c1", "score": 0.9}),
        LCDocument(page_content="B", metadata={"source": "x.md", "chunk_id": "c2", "score": 0.8}),
    ]
    mock_ret = _make_mock_lc_retriever(docs)
    search = LangChainSearchTool(mock_ret)
    tool = search.as_tool()

    await tool.ainvoke({"query": "test"})

    assert search.last_sources == ["x.md"]
    assert search.last_ranked_sources == ["x.md", "x.md"]


async def test_search_tool_handles_no_results():
    mock_ret = _make_mock_lc_retriever(docs=[])
    search = LangChainSearchTool(mock_ret)
    tool = search.as_tool()

    result = await tool.ainvoke({"query": "nothing"})
    assert "No relevant documents found" in result
    assert search.last_ranked_sources == []


async def test_search_tool_accumulates_across_multiple_calls():
    """If the agent calls search twice in one turn, metadata accumulates."""
    docs1 = [
        LCDocument(page_content="A", metadata={"source": "a.md", "chunk_id": "c1", "score": 0.9}),
    ]
    docs2 = [
        LCDocument(page_content="B", metadata={"source": "b.md", "chunk_id": "c2", "score": 0.8}),
    ]
    mock_ret = MagicMock()
    mock_ret.ainvoke = AsyncMock(side_effect=[docs1, docs2])

    search = LangChainSearchTool(mock_ret)
    tool = search.as_tool()

    await tool.ainvoke({"query": "first"})
    await tool.ainvoke({"query": "second"})

    assert search.last_ranked_sources == ["a.md", "b.md"]
    assert search.last_source_chunks == ["A", "B"]
    assert search.last_sources == ["a.md", "b.md"]


async def test_search_tool_reset_clears_state():
    mock_ret = _make_mock_lc_retriever()
    search = LangChainSearchTool(mock_ret)
    tool = search.as_tool()

    await tool.ainvoke({"query": "test"})
    assert len(search.last_ranked_sources) > 0

    search.reset()
    assert search.last_ranked_sources == []
    assert search.last_source_chunks == []
    assert search.last_sources == []


# --- Calculator tool ---


async def test_calculator_evaluates_expression():
    tool = create_calculator_tool()
    result = await tool.ainvoke({"expression": "2 + 3 * 4"})
    assert "14" in result


async def test_calculator_handles_invalid_expression():
    tool = create_calculator_tool()
    result = await tool.ainvoke({"expression": "not_a_number"})
    assert "Error" in result or "error" in result
