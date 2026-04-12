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


# --- Multi-corpus test app (shared across routing / meta / prompt tests) ---


class _FakeOpenAI(MockProvider):
    """Distinct MockProvider subclass so tests can tell it apart from
    the default mock when asserting which orchestrator actually ran."""


@pytest.fixture
def two_corpus_two_provider_app(tmp_path, monkeypatch):
    """Two corpora (fastapi, k8s) × two providers (mock, openai-faked).

    After building the app, each corpus × provider cell gets a *unique*
    MockProvider instance tagged with `_tag`. create_app deliberately
    shares one provider instance across corpora in production (providers
    hold LLM clients and are expensive), but the test needs to distinguish
    which cell ran a given request — so the fixture breaks the sharing
    here and only here.
    """
    from agent_bench.core import provider as provider_mod
    from agent_bench.core.config import (
        AppConfig,
        CorpusConfig,
        EmbeddingConfig,
        ProviderConfig,
        RAGConfig,
        SecurityConfig,
    )
    from agent_bench.serving.app import create_app

    monkeypatch.setattr(provider_mod, "OpenAIProvider", lambda _cfg: _FakeOpenAI())
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = AppConfig(
        provider=ProviderConfig(default="mock"),
        rag=RAGConfig(store_path=str(tmp_path / "store_default")),
        embedding=EmbeddingConfig(cache_dir=str(tmp_path / "emb_cache")),
        security=SecurityConfig(),
        corpora={
            "fastapi": CorpusConfig(
                label="FastAPI Docs",
                store_path=str(tmp_path / "store_fastapi"),
                data_path="data/tech_docs",
            ),
            "k8s": CorpusConfig(
                label="Kubernetes",
                store_path=str(tmp_path / "store_k8s"),
                data_path="data/k8s_docs",
            ),
        },
        default_corpus="fastapi",
    )
    app = create_app(config)

    # Stamp a unique provider into each cell so call_count is per-cell.
    for c_name, inner in app.state.corpus_map.items():
        for p_name, orch in inner.items():
            unique = MockProvider()
            unique._tag = f"{c_name}:{p_name}"  # type: ignore[attr-defined]
            orch.provider = unique
    # Keep the flat orchestrators dict and the singular orchestrator in
    # sync with the per-cell instances for the default corpus.
    app.state.orchestrators = dict(app.state.corpus_map[config.default_corpus])
    app.state.orchestrator = app.state.orchestrators[config.provider.default]
    return app
