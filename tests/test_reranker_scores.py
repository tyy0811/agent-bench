"""Tests for reranker score exposure and retrieval metadata threading."""

import numpy as np
import pytest

from agent_bench.rag.chunker import Chunk
from agent_bench.rag.reranker import CrossEncoderReranker
from agent_bench.rag.retriever import Retriever

SAMPLE_CHUNKS = [
    Chunk(id=f"c{i}", content=f"Content about topic {i}", source=f"doc_{i}.md",
          chunk_index=0, metadata={})
    for i in range(5)
]


class MockCrossEncoder:
    """Deterministic cross-encoder returning predictable scores."""
    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        # Score = inverse of chunk index (c0 gets highest)
        return np.array([5.0 - i for i in range(len(pairs))])


class TestRerankerScores:
    def test_rerank_returns_chunk_score_tuples(self):
        reranker = CrossEncoderReranker(model=MockCrossEncoder())
        results = reranker.rerank("test query", SAMPLE_CHUNKS, top_k=3)

        assert len(results) == 3
        for item in results:
            assert isinstance(item, tuple)
            assert isinstance(item[0], Chunk)
            assert isinstance(item[1], float)

    def test_rerank_scores_are_cross_encoder_scores(self):
        reranker = CrossEncoderReranker(model=MockCrossEncoder())
        results = reranker.rerank("test query", SAMPLE_CHUNKS, top_k=3)

        # MockCrossEncoder gives 5.0, 4.0, 3.0, 2.0, 1.0 — top 3 are 5.0, 4.0, 3.0
        chunks, scores = zip(*results)
        assert scores == (5.0, 4.0, 3.0)

    def test_rerank_sorted_descending(self):
        reranker = CrossEncoderReranker(model=MockCrossEncoder())
        results = reranker.rerank("test query", SAMPLE_CHUNKS, top_k=5)

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_empty_input(self):
        reranker = CrossEncoderReranker(model=MockCrossEncoder())
        results = reranker.rerank("test query", [], top_k=3)
        assert results == []


class TestRetrieverScoreThreading:
    @pytest.mark.asyncio
    async def test_retriever_sets_rerank_score(self, mock_embedder, test_store):
        reranker = CrossEncoderReranker(model=MockCrossEncoder())
        retriever = Retriever(
            embedder=mock_embedder, store=test_store,
            reranker=reranker, reranker_top_k=3,
        )
        result = await retriever.search("path parameters", top_k=5)

        assert result.pre_rerank_count > 0
        for r in result.results:
            assert r.rerank_score is not None

    @pytest.mark.asyncio
    async def test_retriever_without_reranker_has_no_rerank_score(self, mock_embedder, test_store):
        retriever = Retriever(embedder=mock_embedder, store=test_store)
        result = await retriever.search("path parameters", top_k=3)

        assert result.pre_rerank_count == 0
        for r in result.results:
            assert r.rerank_score is None
