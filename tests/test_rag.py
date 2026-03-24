"""Tests for RAG pipeline: chunker, embedder, store, and retriever."""

from __future__ import annotations

import numpy as np
import pytest

from agent_bench.rag.chunker import chunk_fixed, chunk_recursive, chunk_text
from agent_bench.rag.embedder import Embedder
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
        results = await test_retriever.search("path parameters", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_strategy_override(self, test_retriever: Retriever):
        results = await test_retriever.search("Pydantic models", top_k=3, strategy="keyword")
        assert len(results) > 0
        assert all(r.retrieval_strategy == "keyword" for r in results)
