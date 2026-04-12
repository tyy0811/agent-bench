"""Tests for per-request corpus routing.

Exercises the full corpus × provider matrix through /ask and /ask/stream.
Uses create_app with a multi-corpus fixture and monkeypatches a second
provider so both toggles can be tested without real API keys.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from agent_bench.core.config import (
    AppConfig,
    CorpusConfig,
    EmbeddingConfig,
    ProviderConfig,
    RAGConfig,
    SecurityConfig,
)
from agent_bench.core.provider import MockProvider
from agent_bench.serving.app import create_app


class _FakeOpenAI(MockProvider):
    """Distinct MockProvider subclass so we can distinguish it from the
    default mock when asserting which orchestrator actually ran."""


@pytest.fixture
def two_corpus_two_provider_app(tmp_path, monkeypatch):
    """Two corpora (fastapi, k8s) × two providers (mock, openai-faked).

    After building the app, each corpus×provider cell gets a *unique*
    MockProvider instance stamped with a `_tag` attribute. create_app
    deliberately shares one provider instance across corpora (it's an
    expensive object), but the test needs to distinguish which cell ran
    a given request, so we break the sharing here and only here.
    """
    from agent_bench.core import provider as provider_mod

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


def _reset_call_counts(app):
    """Zero out provider.call_count on every orchestrator in corpus_map."""
    for inner in app.state.corpus_map.values():
        for orch in inner.values():
            orch.provider.call_count = 0


class TestCorpusRouting:
    @pytest.mark.asyncio
    async def test_default_corpus_default_provider(self, two_corpus_two_provider_app):
        app = two_corpus_two_provider_app
        _reset_call_counts(app)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post("/ask", json={"question": "hello"})
        assert resp.status_code == 200
        # Default corpus × default provider should have been called.
        assert app.state.corpus_map["fastapi"]["mock"].provider.call_count > 0
        # Nothing else.
        assert app.state.corpus_map["fastapi"]["openai"].provider.call_count == 0
        assert app.state.corpus_map["k8s"]["mock"].provider.call_count == 0
        assert app.state.corpus_map["k8s"]["openai"].provider.call_count == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("corpus,provider", [
        ("fastapi", "mock"),
        ("fastapi", "openai"),
        ("k8s", "mock"),
        ("k8s", "openai"),
    ])
    async def test_full_routing_matrix(
        self, two_corpus_two_provider_app, corpus, provider,
    ):
        """Every (corpus, provider) pair routes to its own orchestrator."""
        app = two_corpus_two_provider_app
        _reset_call_counts(app)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/ask",
                json={"question": "hi", "corpus": corpus, "provider": provider},
            )
        assert resp.status_code == 200
        target = app.state.corpus_map[corpus][provider]
        assert target.provider.call_count > 0, (
            f"expected {corpus}×{provider} orchestrator to be called"
        )
        # No other corpus×provider cell was touched.
        for c_name, inner in app.state.corpus_map.items():
            for p_name, orch in inner.items():
                if (c_name, p_name) != (corpus, provider):
                    assert orch.provider.call_count == 0, (
                        f"unexpected call on {c_name}×{p_name}"
                    )

    @pytest.mark.asyncio
    async def test_unknown_corpus_returns_422(self, two_corpus_two_provider_app):
        app = two_corpus_two_provider_app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/ask", json={"question": "hi", "corpus": "eu_ai_act"},
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_stream_endpoint_also_routes(self, two_corpus_two_provider_app):
        """/ask/stream follows the same routing path as /ask."""
        app = two_corpus_two_provider_app
        _reset_call_counts(app)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/ask/stream",
                json={"question": "hi", "corpus": "k8s", "provider": "openai"},
            )
        assert resp.status_code == 200
        assert app.state.corpus_map["k8s"]["openai"].provider.call_count > 0
        assert app.state.corpus_map["fastapi"]["mock"].provider.call_count == 0


class TestLegacyRouting:
    """Without corpora configured, /ask uses the flat orchestrators dict."""

    @pytest.fixture
    def legacy_app(self, tmp_path):
        config = AppConfig(
            provider=ProviderConfig(default="mock"),
            rag=RAGConfig(store_path=str(tmp_path / "store")),
            embedding=EmbeddingConfig(cache_dir=str(tmp_path / "emb")),
            security=SecurityConfig(),
        )
        return create_app(config)

    @pytest.mark.asyncio
    async def test_legacy_request_without_corpus_field(self, legacy_app):
        async with AsyncClient(
            transport=ASGITransport(app=legacy_app), base_url="http://test",
        ) as client:
            resp = await client.post("/ask", json={"question": "hi"})
        assert resp.status_code == 200
        # corpus_map is empty in legacy mode
        assert legacy_app.state.corpus_map == {}

    @pytest.mark.asyncio
    async def test_legacy_rejects_corpus_field(self, legacy_app):
        """Even in legacy mode the Literal validator still fires."""
        async with AsyncClient(
            transport=ASGITransport(app=legacy_app), base_url="http://test",
        ) as client:
            # A valid Literal value but corpus_map is empty — should still
            # succeed via legacy fallback.
            resp = await client.post(
                "/ask", json={"question": "hi", "corpus": "fastapi"},
            )
        assert resp.status_code == 200
