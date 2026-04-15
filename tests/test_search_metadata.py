"""Tests for enriched SearchTool metadata used by SSE stage events."""

import pytest

from agent_bench.rag.chunker import Chunk
from agent_bench.rag.retriever import RetrievalResult
from agent_bench.rag.store import SearchResult
from agent_bench.tools.search import SearchTool


class FakeRetriever:
    """Returns canned RetrievalResult with known scores and previews."""
    async def search(self, query, top_k=5, strategy=None):
        chunks = [
            SearchResult(
                chunk=Chunk(id=f"c{i}", content=f"Content about topic {i} " * 20,
                           source=f"doc_{i}.md", chunk_index=0, metadata={}),
                score=0.5 - i * 0.1,
                rank=i + 1,
                retrieval_strategy="hybrid+reranker",
                rerank_score=0.9 - i * 0.1,
            )
            for i in range(3)
        ]
        return RetrievalResult(results=chunks, pre_rerank_count=10)


class FakeRetrieverWithPII:
    async def search(self, query, top_k=5, strategy=None):
        chunks = [
            SearchResult(
                chunk=Chunk(id="c0", content="Contact john@example.com for help",
                           source="doc.md", chunk_index=0, metadata={}),
                score=0.5, rank=1, retrieval_strategy="hybrid",
            ),
        ]
        return RetrievalResult(results=chunks, pre_rerank_count=0)


class TestSearchToolMetadata:
    @pytest.mark.asyncio
    async def test_metadata_includes_pre_rerank_count(self):
        tool = SearchTool(retriever=FakeRetriever(), refusal_threshold=0.0)
        output = await tool.execute(query="test")
        assert output.metadata["pre_rerank_count"] == 10

    @pytest.mark.asyncio
    async def test_metadata_includes_chunks_with_scores_and_previews(self):
        tool = SearchTool(retriever=FakeRetriever(), refusal_threshold=0.0)
        output = await tool.execute(query="test")

        chunks = output.metadata["chunks"]
        assert len(chunks) == 3
        for chunk in chunks:
            assert "source" in chunk
            assert "score" in chunk
            assert "preview" in chunk
            assert len(chunk["preview"]) <= 120

    @pytest.mark.asyncio
    async def test_metadata_includes_pii_count_zero_when_no_redactor(self):
        tool = SearchTool(retriever=FakeRetriever(), refusal_threshold=0.0)
        output = await tool.execute(query="test")
        assert output.metadata["pii_redactions_count"] == 0

    @pytest.mark.asyncio
    async def test_metadata_includes_pii_count_with_redactor(self):
        from agent_bench.security.pii_redactor import PIIRedactor

        redactor = PIIRedactor(mode="redact")
        retriever = FakeRetrieverWithPII()
        tool = SearchTool(retriever=retriever, refusal_threshold=0.0, pii_redactor=redactor)
        output = await tool.execute(query="test")
        assert output.metadata["pii_redactions_count"] > 0

    @pytest.mark.asyncio
    async def test_refusal_metadata_includes_threshold(self):
        tool = SearchTool(retriever=FakeRetriever(), refusal_threshold=0.8)
        output = await tool.execute(query="test")
        assert output.metadata.get("refused") is True
        assert output.metadata["refusal_threshold"] == 0.8
        assert "max_score" in output.metadata
