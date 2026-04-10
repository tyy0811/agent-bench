"""Tests for RAG pipeline: chunker, embedder, store, retriever, and reranker."""

from __future__ import annotations

import numpy as np
import pytest

from agent_bench.rag.chunker import Chunk, chunk_fixed, chunk_recursive, chunk_text
from agent_bench.rag.embedder import Embedder
from agent_bench.rag.reranker import CrossEncoderReranker
from agent_bench.rag.retriever import Retriever
from agent_bench.rag.store import HybridStore, SearchResult

# --- Chunker tests ---


class TestChunker:
    SAMPLE_TEXT = (
        "FastAPI is a modern web framework.\n\n"
        "It is based on standard Python type hints.\n\n"
        "Path parameters are declared in the URL path using curly braces. "
        "You can specify their types using annotations.\n\n"
        "Query parameters are parsed automatically from the query string. "
        "They support default values and optional types.\n\n"
        "Request bodies use Pydantic models for validation."
    )

    def test_recursive_within_size_limits(self):
        chunk_size = 100
        overlap = 64
        chunks = chunk_recursive(
            self.SAMPLE_TEXT, "test.md", chunk_size=chunk_size, chunk_overlap=overlap
        )
        for c in chunks:
            # Overlap prepend may push up to overlap chars beyond chunk_size
            assert len(c.content) <= chunk_size + overlap + 1, (
                f"Chunk too long: {len(c.content)} chars"
            )
        assert len(chunks) > 1

    def test_fixed_within_size_limits(self):
        chunks = chunk_fixed(self.SAMPLE_TEXT, "test.md", chunk_size=100, chunk_overlap=20)
        for c in chunks:
            assert len(c.content) <= 100
        assert len(chunks) > 1

    def test_recursive_preserves_text(self):
        """Every word in the source should appear in at least one chunk."""
        chunks = chunk_recursive(self.SAMPLE_TEXT, "test.md", chunk_size=200)
        all_words = set(self.SAMPLE_TEXT.split())
        chunk_words = set()
        for c in chunks:
            chunk_words.update(c.content.split())
        assert all_words.issubset(chunk_words)

    def test_fixed_preserves_text_coverage(self):
        """Every word in the source should appear in at least one chunk."""
        chunks = chunk_fixed(self.SAMPLE_TEXT, "test.md", chunk_size=100, chunk_overlap=20)
        all_words = set(self.SAMPLE_TEXT.split())
        chunk_words = set()
        for c in chunks:
            chunk_words.update(c.content.split())
        assert all_words.issubset(chunk_words)

    def test_chunk_source_is_bare_filename(self):
        chunks = chunk_text(self.SAMPLE_TEXT, "fastapi_intro.md", strategy="recursive")
        for c in chunks:
            assert c.source == "fastapi_intro.md"
            assert "/" not in c.source

    def test_chunk_text_dispatcher(self):
        rec = chunk_text(self.SAMPLE_TEXT, "t.md", strategy="recursive", chunk_size=200)
        fix = chunk_text(self.SAMPLE_TEXT, "t.md", strategy="fixed", chunk_size=200)
        assert all(c.metadata.get("strategy") == "recursive" for c in rec)
        assert all(c.metadata.get("strategy") == "fixed" for c in fix)

    def test_empty_text(self):
        assert chunk_recursive("", "empty.md") == []
        assert chunk_fixed("", "empty.md") == []


# --- Embedder tests ---


class TestEmbedder:
    def test_embed_produces_correct_shape(self, mock_embedder: Embedder):
        vec = mock_embedder.embed("test sentence")
        assert vec.shape == (384,)

    def test_embed_is_normalized(self, mock_embedder: Embedder):
        vec = mock_embedder.embed("test sentence")
        norm = np.linalg.norm(vec)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_embed_batch_shape(self, mock_embedder: Embedder):
        vecs = mock_embedder.embed_batch(["sentence one", "sentence two", "sentence three"])
        assert vecs.shape == (3, 384)

    def test_cache_hit_skips_model(self, mock_embedding_model, tmp_path):
        """Second embed() call for same text should use cache, not model."""
        embedder = Embedder(model=mock_embedding_model, cache_dir=str(tmp_path))
        _ = embedder.embed("cache test")
        calls_after_first = mock_embedding_model.call_count
        _ = embedder.embed("cache test")
        assert mock_embedding_model.call_count == calls_after_first

    def test_different_texts_produce_different_embeddings(self, mock_embedder: Embedder):
        v1 = mock_embedder.embed("path parameters")
        v2 = mock_embedder.embed("query parameters")
        assert not np.allclose(v1, v2)


# --- Store tests ---


class TestHybridStore:
    def test_add_and_semantic_search(self, test_store: HybridStore, mock_embedder: Embedder):
        """Semantic search returns relevant result for a known query."""
        query_vec = mock_embedder.embed("path parameters curly braces")
        results = test_store.search(
            query_embedding=query_vec,
            query_text="path parameters curly braces",
            top_k=3,
            strategy="semantic",
        )
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        # Should have scores and ranks
        assert results[0].rank == 1
        assert results[0].retrieval_strategy == "semantic"

    def test_keyword_search(self, test_store: HybridStore, mock_embedder: Embedder):
        """BM25 keyword search finds chunks with matching terms."""
        query_vec = mock_embedder.embed("Pydantic models validation")
        results = test_store.search(
            query_embedding=query_vec,
            query_text="Pydantic models validation",
            top_k=3,
            strategy="keyword",
        )
        assert len(results) > 0
        # Top result should be the request body chunk (mentions Pydantic)
        assert "Pydantic" in results[0].chunk.content

    def test_hybrid_returns_results_from_both(
        self, test_store: HybridStore, mock_embedder: Embedder
    ):
        """RRF hybrid search returns results — both dense and sparse contribute."""
        query_vec = mock_embedder.embed("path parameters FastAPI")
        results = test_store.search(
            query_embedding=query_vec,
            query_text="path parameters FastAPI",
            top_k=5,
            strategy="hybrid",
        )
        assert len(results) > 0
        assert all(r.retrieval_strategy == "hybrid" for r in results)
        # RRF scores should be positive and sorted descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_empty_store(self):
        store = HybridStore(dimension=384)
        dummy_vec = np.random.randn(384).astype(np.float32)
        results = store.search(
            query_embedding=dummy_vec, query_text="test", top_k=5, strategy="hybrid"
        )
        assert results == []

    def test_save_load_roundtrip(self, test_store: HybridStore, mock_embedder: Embedder, tmp_path):
        """Save and load preserves all data and produces same search results."""
        store_path = tmp_path / "test_store"

        # Search before save
        query_vec = mock_embedder.embed("path parameters")
        results_before = test_store.search(
            query_embedding=query_vec,
            query_text="path parameters",
            top_k=3,
            strategy="hybrid",
        )

        # Save and reload
        test_store.save(store_path)
        loaded = HybridStore.load(store_path, rrf_k=60)

        # Stats match
        assert loaded.stats().total_chunks == test_store.stats().total_chunks
        assert loaded.stats().faiss_index_size == test_store.stats().faiss_index_size

        # Search after load
        results_after = loaded.search(
            query_embedding=query_vec,
            query_text="path parameters",
            top_k=3,
            strategy="hybrid",
        )
        assert len(results_after) == len(results_before)
        assert [r.chunk.id for r in results_after] == [r.chunk.id for r in results_before]

    def test_stats(self, test_store: HybridStore):
        stats = test_store.stats()
        assert stats.total_chunks == 5
        assert stats.faiss_index_size == 5
        assert stats.unique_sources == 4  # 4 unique source files in sample chunks


# --- Retriever tests ---


class TestRetriever:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, test_retriever: Retriever):
        result = await test_retriever.search("path parameters", top_k=3)
        assert len(result.results) > 0
        assert all(isinstance(r, SearchResult) for r in result.results)

    @pytest.mark.asyncio
    async def test_search_strategy_override(self, test_retriever: Retriever):
        result = await test_retriever.search("Pydantic models", top_k=3, strategy="keyword")
        assert len(result.results) > 0
        assert all(r.retrieval_strategy == "keyword" for r in result.results)


# --- Reranker tests ---


class MockCrossEncoder:
    """Mock cross-encoder that returns deterministic scores based on content length."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        # Score based on content length — longer content scores higher
        # This gives a deterministic, predictable reordering
        return [float(len(content)) for _, content in pairs]


class TestCrossEncoderReranker:
    def _make_chunks(self, contents: list[str]) -> list[Chunk]:
        return [
            Chunk(id=f"c{i}", content=c, source=f"doc_{i}.md", chunk_index=0)
            for i, c in enumerate(contents)
        ]

    def test_reranker_reorders(self):
        """Reranker reorders chunks by cross-encoder score."""
        chunks = self._make_chunks(["short", "a medium length chunk", "longest chunk content here"])
        reranker = CrossEncoderReranker(model=MockCrossEncoder())
        result = reranker.rerank("test query", chunks, top_k=3)

        # MockCrossEncoder scores by content length, so longest first
        assert result[0][0].content == "longest chunk content here"
        assert result[1][0].content == "a medium length chunk"
        assert result[2][0].content == "short"

    def test_reranker_top_k(self):
        """Reranker returns exactly top_k results from a larger input."""
        chunks = self._make_chunks([f"content {i}" for i in range(20)])
        reranker = CrossEncoderReranker(model=MockCrossEncoder())
        result = reranker.rerank("test query", chunks, top_k=5)
        assert len(result) == 5

    def test_reranker_disabled(self, mock_embedder: Embedder, test_store: HybridStore):
        """Retriever without reranker preserves RRF order."""
        retriever_no_reranker = Retriever(embedder=mock_embedder, store=test_store)
        retriever_with_none = Retriever(
            embedder=mock_embedder, store=test_store, reranker=None,
        )

        import asyncio

        results_a = asyncio.get_event_loop().run_until_complete(
            retriever_no_reranker.search("path parameters", top_k=3)
        )
        results_b = asyncio.get_event_loop().run_until_complete(
            retriever_with_none.search("path parameters", top_k=3)
        )
        assert [r.chunk.id for r in results_a.results] == [r.chunk.id for r in results_b.results]

    def test_reranker_empty_input(self):
        """Empty chunk list returns empty list."""
        reranker = CrossEncoderReranker(model=MockCrossEncoder())
        result = reranker.rerank("test query", [], top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_reranked_results_preserve_rrf_scores(
        self, mock_embedder: Embedder, test_store: HybridStore,
    ):
        """Reranked results carry original RRF scores, not 0.0.

        This is critical: the refusal gate in SearchTool checks max_score
        from the returned results. If reranking zeroes out scores, the
        refusal gate would reject every reranked query.
        """
        reranker = CrossEncoderReranker(model=MockCrossEncoder())
        retriever = Retriever(
            embedder=mock_embedder,
            store=test_store,
            reranker=reranker,
            reranker_top_k=3,
        )
        result = await retriever.search("path parameters", top_k=3)
        assert len(result.results) > 0
        # All scores must be positive (preserved from RRF), not 0.0
        scores = [r.score for r in result.results]
        assert all(r.score > 0 for r in result.results), (
            f"Reranked scores should be positive RRF scores, got: {scores}"
        )

    @pytest.mark.asyncio
    async def test_refusal_with_reranker_enabled(self):
        """Integration: out-of-scope query with reranker on still refuses.

        The refusal gate fires on RRF max_score BEFORE reranking (go/no-go
        decision). This test validates the Feature 1 + Feature 2 interaction.
        """
        from agent_bench.tools.search import SearchTool
        from tests.test_tools import MockChunk, MockRetriever, MockSearchResult

        # Low scores — should trigger refusal regardless of reranker
        low_score_results = [
            MockSearchResult(
                chunk=MockChunk(content="Unrelated content", source="irrelevant.md"),
                score=0.005,
            ),
        ]
        retriever = MockRetriever(results=low_score_results)
        tool = SearchTool(retriever=retriever, refusal_threshold=0.02)
        result = await tool.execute(query="how to cook pasta")

        assert result.metadata["refused"] is True
        assert "No relevant documents found" in result.result
