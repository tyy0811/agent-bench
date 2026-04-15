"""Tests for per-request corpus routing.

Exercises the full corpus × provider matrix through /ask and /ask/stream.
The multi-corpus test-app fixture (`two_corpus_two_provider_app`) lives
in tests/conftest.py and is shared with test_meta_corpus.py and
test_prompt_template.py.
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
from agent_bench.serving.app import create_app


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


class TestMisconfiguredCorpus:
    """body.corpus valid-per-Literal but not in corpus_map should fail
    loud at request time with 400 instead of silently falling through
    to the legacy orchestrator."""

    @pytest.fixture
    def fastapi_only_app(self, tmp_path, monkeypatch):
        """Multi-corpus mode with ONLY fastapi configured (k8s removed)."""
        from agent_bench.core import provider as provider_mod
        from agent_bench.core.provider import MockProvider

        monkeypatch.setattr(
            provider_mod, "OpenAIProvider", lambda _cfg: MockProvider(),
        )
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
            },
            default_corpus="fastapi",
        )
        return create_app(config)

    @pytest.mark.asyncio
    async def test_unconfigured_corpus_returns_400(self, fastapi_only_app):
        """k8s passes Literal but is not in corpus_map — expect 400."""
        async with AsyncClient(
            transport=ASGITransport(app=fastapi_only_app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/ask", json={"question": "hi", "corpus": "k8s"},
            )
        assert resp.status_code == 400
        detail = resp.json().get("detail", "")
        assert "not configured" in detail.lower()
        assert "k8s" in detail
        assert "fastapi" in detail  # lists available corpora

    @pytest.mark.asyncio
    async def test_unconfigured_corpus_returns_400_stream(self, fastapi_only_app):
        """Same guard on /ask/stream."""
        async with AsyncClient(
            transport=ASGITransport(app=fastapi_only_app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/ask/stream", json={"question": "hi", "corpus": "k8s"},
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_default_corpus_still_works(self, fastapi_only_app):
        """fastapi (the only configured corpus) still routes fine."""
        async with AsyncClient(
            transport=ASGITransport(app=fastapi_only_app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/ask", json={"question": "hi", "corpus": "fastapi"},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_unavailable_provider_returns_400(self, fastapi_only_app):
        """Explicitly asking for a provider not wired on the server must
        fail loud with 400 instead of silently running on the default.

        The fastapi_only_app fixture only has 'mock' wired (no
        OPENAI_API_KEY is set, so the openai alt provider is never
        added to providers dict). A dashboard request with
        provider='anthropic' used to silently run on mock."""
        async with AsyncClient(
            transport=ASGITransport(app=fastapi_only_app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/ask",
                json={
                    "question": "hi",
                    "corpus": "fastapi",
                    "provider": "anthropic",
                },
            )
        assert resp.status_code == 400
        detail = resp.json().get("detail", "")
        assert "anthropic" in detail.lower()
        assert "not available" in detail.lower()


class TestResolveOrchestratorDirect:
    """Unit tests for _resolve_orchestrator without the HTTP stack.

    Builds a fake Request object with just the app.state attributes the
    helper reads. Catches edge cases that integration tests would miss
    (unknown provider, explicit provider not in inner dict, etc.).
    """

    @pytest.fixture
    def fake_request_builder(self):
        """Return a factory that makes a fake Request with the given state."""
        from types import SimpleNamespace

        def build(
            corpus_map,
            default_corpus,
            provider_default,
            orchestrators=None,
            orchestrator=None,
            system_prompt="legacy prompt",
        ):
            state = SimpleNamespace(
                config=SimpleNamespace(
                    corpora={k: SimpleNamespace(label=k.title()) for k in corpus_map},
                    default_corpus=default_corpus,
                    provider=SimpleNamespace(default=provider_default),
                ),
                corpus_map=corpus_map,
                orchestrators=orchestrators or {},
                orchestrator=orchestrator,
                system_prompt=system_prompt,
            )
            return SimpleNamespace(app=SimpleNamespace(state=state))

        return build

    def _make_body(self, corpus=None, provider=None):
        from agent_bench.serving.schemas import AskRequest

        return AskRequest(question="x", corpus=corpus, provider=provider)

    def test_multi_corpus_happy_path(self, fake_request_builder):
        from agent_bench.serving.routes import _resolve_orchestrator

        sentinel = object()
        req = fake_request_builder(
            corpus_map={"fastapi": {"mock": sentinel}},
            default_corpus="fastapi",
            provider_default="mock",
        )
        orch, name, provider_name = _resolve_orchestrator(req, self._make_body())
        assert orch is sentinel
        assert name == "fastapi"
        assert provider_name == "mock"

    def test_explicit_provider_routes_correctly(self, fake_request_builder):
        """body.provider=openai picks the openai cell, not a fallback."""
        from agent_bench.serving.routes import _resolve_orchestrator

        mock_sent = object()
        oai_sent = object()
        req = fake_request_builder(
            corpus_map={"fastapi": {"mock": mock_sent, "openai": oai_sent}},
            default_corpus="fastapi",
            provider_default="mock",
        )
        orch, _, provider_name = _resolve_orchestrator(
            req, self._make_body(provider="openai"),
        )
        assert orch is oai_sent
        assert provider_name == "openai"
        # Implicit provider uses corpus default.
        orch, _, provider_name = _resolve_orchestrator(req, self._make_body())
        assert orch is mock_sent
        assert provider_name == "mock"

    def test_explicit_unavailable_provider_raises_400(self, fake_request_builder):
        """body.provider explicitly names a provider not wired for the
        corpus — fail closed with 400 instead of silently falling back."""
        from fastapi import HTTPException

        from agent_bench.serving.routes import _resolve_orchestrator

        req = fake_request_builder(
            # Only mock is wired — no openai, no anthropic.
            corpus_map={"fastapi": {"mock": object()}},
            default_corpus="fastapi",
            provider_default="mock",
        )
        with pytest.raises(HTTPException) as exc_info:
            _resolve_orchestrator(req, self._make_body(provider="anthropic"))
        assert exc_info.value.status_code == 400
        assert "anthropic" in exc_info.value.detail
        assert "mock" in exc_info.value.detail  # lists what IS available

    def test_explicit_unconfigured_corpus_raises_400(self, fake_request_builder):
        from fastapi import HTTPException

        from agent_bench.serving.routes import _resolve_orchestrator

        req = fake_request_builder(
            corpus_map={"fastapi": {"mock": object()}},
            default_corpus="fastapi",
            provider_default="mock",
        )
        with pytest.raises(HTTPException) as exc_info:
            _resolve_orchestrator(req, self._make_body(corpus="k8s"))
        assert exc_info.value.status_code == 400
        assert "k8s" in exc_info.value.detail
        assert "fastapi" in exc_info.value.detail

    def test_legacy_mode_uses_flat_orchestrators(self, fake_request_builder):
        from agent_bench.serving.routes import _resolve_orchestrator

        legacy_orch = object()
        flat_oai = object()
        req = fake_request_builder(
            corpus_map={},
            default_corpus="",
            provider_default="mock",
            orchestrators={"openai": flat_oai},
            orchestrator=legacy_orch,
        )
        # body.provider=openai finds it in flat dict
        orch, _, provider_name = _resolve_orchestrator(
            req, self._make_body(provider="openai"),
        )
        assert orch is flat_oai
        assert provider_name == "openai"
        # No provider falls back to app.state.orchestrator
        orch, _, provider_name = _resolve_orchestrator(req, self._make_body())
        assert orch is legacy_orch
        assert provider_name == "mock"

    def test_legacy_mode_explicit_unavailable_provider_raises_400(
        self, fake_request_builder,
    ):
        """Legacy mode is also strict on explicit unavailable providers."""
        from fastapi import HTTPException

        from agent_bench.serving.routes import _resolve_orchestrator

        req = fake_request_builder(
            corpus_map={},
            default_corpus="",
            provider_default="mock",
            orchestrators={"mock": object()},  # only mock wired
            orchestrator=object(),
        )
        with pytest.raises(HTTPException) as exc_info:
            _resolve_orchestrator(req, self._make_body(provider="anthropic"))
        assert exc_info.value.status_code == 400
        assert "anthropic" in exc_info.value.detail


class TestResolveSystemPromptDirect:
    """Unit tests for _resolve_system_prompt."""

    def _build_req(self, corpora, system_prompt="legacy"):
        from types import SimpleNamespace

        state = SimpleNamespace(
            config=SimpleNamespace(corpora=corpora),
            system_prompt=system_prompt,
        )
        return SimpleNamespace(app=SimpleNamespace(state=state))

    def test_multi_corpus_formats_template(self):
        from types import SimpleNamespace

        from agent_bench.serving.routes import _resolve_system_prompt

        req = self._build_req(
            {"fastapi": SimpleNamespace(label="FastAPI Docs")},
        )
        prompt, label = _resolve_system_prompt(req, "fastapi")
        assert label == "FastAPI Docs"
        assert "FastAPI Docs" in prompt
        assert "{corpus_label}" not in prompt
        assert "refuse" in prompt.lower()

    def test_legacy_returns_task_prompt(self):
        from agent_bench.serving.routes import _resolve_system_prompt

        req = self._build_req({}, system_prompt="legacy task prompt")
        prompt, label = _resolve_system_prompt(req, "")
        assert prompt == "legacy task prompt"
        assert label == ""

    def test_unknown_corpus_name_falls_to_legacy(self):
        """If corpus_name isn't in corpora (shouldn't happen post-resolve
        because of the 400 guard, but the helper should still be safe)."""
        from types import SimpleNamespace

        from agent_bench.serving.routes import _resolve_system_prompt

        req = self._build_req(
            {"fastapi": SimpleNamespace(label="FastAPI Docs")},
            system_prompt="legacy",
        )
        prompt, label = _resolve_system_prompt(req, "nonexistent")
        assert prompt == "legacy"
        assert label == ""
