"""Tests for LangChain retriever wrapper around agent-bench's async Retriever."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_bench.langchain_baseline.retriever import AgentBenchRetriever


def _make_mock_retriever(results=None):
    """Create a mock of agent_bench.rag.retriever.Retriever."""
    retriever = MagicMock()
    if results is None:
        # Default: one result with known fields
        result = MagicMock()
        result.chunk.content = "Path parameters use curly braces."
        result.chunk.source = "fastapi_path_params.md"
        result.chunk.id = "chunk_001"
        result.score = 0.85
        result.rank = 1
        results = [result]
    retriever.search = AsyncMock(return_value=results)
    return retriever


async def test_returns_langchain_documents():
    mock_ret = _make_mock_retriever()
    wrapper = AgentBenchRetriever(retriever=mock_ret, top_k=5)
    docs = await wrapper.ainvoke("path parameters")

    assert len(docs) == 1
    assert docs[0].page_content == "Path parameters use curly braces."
    assert docs[0].metadata["source"] == "fastapi_path_params.md"
    assert docs[0].metadata["chunk_id"] == "chunk_001"
    assert docs[0].metadata["score"] == 0.85


async def test_passes_top_k_to_underlying_retriever():
    mock_ret = _make_mock_retriever()
    wrapper = AgentBenchRetriever(retriever=mock_ret, top_k=3)
    await wrapper.ainvoke("test")
    mock_ret.search.assert_called_once_with("test", top_k=3)


async def test_handles_empty_results():
    mock_ret = _make_mock_retriever(results=[])
    wrapper = AgentBenchRetriever(retriever=mock_ret, top_k=5)
    docs = await wrapper.ainvoke("nonsense")
    assert docs == []


async def test_multiple_results_preserve_order():
    r1 = MagicMock()
    r1.chunk.content = "First"
    r1.chunk.source = "a.md"
    r1.chunk.id = "c1"
    r1.score = 0.9

    r2 = MagicMock()
    r2.chunk.content = "Second"
    r2.chunk.source = "b.md"
    r2.chunk.id = "c2"
    r2.score = 0.7

    mock_ret = _make_mock_retriever(results=[r1, r2])
    wrapper = AgentBenchRetriever(retriever=mock_ret, top_k=5)
    docs = await wrapper.ainvoke("test")

    assert len(docs) == 2
    assert docs[0].page_content == "First"
    assert docs[1].page_content == "Second"
