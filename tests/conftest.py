"""Shared test fixtures."""

import numpy as np
import pytest

from agent_bench.core.provider import MockProvider
from agent_bench.rag.chunker import Chunk
from agent_bench.rag.embedder import Embedder
from agent_bench.rag.retriever import Retriever
from agent_bench.rag.store import HybridStore


@pytest.fixture
def mock_provider() -> MockProvider:
    """MockProvider instance for deterministic testing."""
    return MockProvider()


class MockEmbeddingModel:
    """Deterministic embedding model for tests. No model download needed.

    Uses seeded random vectors, normalized to unit length.
    Same input always produces the same output via content hashing.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.call_count = 0

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.call_count += 1
        vecs = []
        for s in sentences:
            seed = int.from_bytes(s.encode()[:4], "big") % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dimension).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vecs.append(vec)
        return np.stack(vecs)


@pytest.fixture
def mock_embedding_model() -> MockEmbeddingModel:
    """Deterministic embedding model — no model download."""
    return MockEmbeddingModel()


@pytest.fixture
def mock_embedder(mock_embedding_model: MockEmbeddingModel, tmp_path: object) -> Embedder:
    """Embedder backed by mock model with temp cache dir."""
    return Embedder(model=mock_embedding_model, cache_dir=str(tmp_path))


SAMPLE_CHUNKS = [
    Chunk(
        id="chunk_path_1",
        content="Path parameters in FastAPI are defined using curly braces in the URL path.",
        source="fastapi_path_params.md",
        chunk_index=0,
        metadata={"strategy": "recursive"},
    ),
    Chunk(
        id="chunk_path_2",
        content="You can declare the type of a path parameter using Python type annotations.",
        source="fastapi_path_params.md",
        chunk_index=1,
        metadata={"strategy": "recursive"},
    ),
    Chunk(
        id="chunk_query_1",
        content="Query parameters are automatically parsed from the URL query string.",
        source="fastapi_query_params.md",
        chunk_index=0,
        metadata={"strategy": "recursive"},
    ),
    Chunk(
        id="chunk_body_1",
        content="Request body data is defined using Pydantic models in FastAPI.",
        source="fastapi_request_body.md",
        chunk_index=0,
        metadata={"strategy": "recursive"},
    ),
    Chunk(
        id="chunk_response_1",
        content="Response models control the output schema of your API endpoints.",
        source="fastapi_response_model.md",
        chunk_index=0,
        metadata={"strategy": "recursive"},
    ),
]


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """5 sample chunks with known content and sources."""
    return list(SAMPLE_CHUNKS)


@pytest.fixture
def test_store(mock_embedder: Embedder, sample_chunks: list[Chunk]) -> HybridStore:
    """HybridStore populated with sample chunks via mock embedder."""
    store = HybridStore(dimension=384, rrf_k=60)
    texts = [c.content for c in sample_chunks]
    embeddings = mock_embedder.embed_batch(texts)
    store.add(sample_chunks, embeddings)
    return store


@pytest.fixture
def test_retriever(mock_embedder: Embedder, test_store: HybridStore) -> Retriever:
    """Retriever wired to mock embedder + test store."""
    return Retriever(embedder=mock_embedder, store=test_store)
