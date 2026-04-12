"""Tests for multi-corpus construction at app startup."""

import pytest

from agent_bench.core.config import (
    AppConfig,
    CorpusConfig,
    EmbeddingConfig,
    ProviderConfig,
    RAGConfig,
)
from agent_bench.serving.app import create_app


@pytest.fixture
def multi_corpus_config(tmp_path):
    """Config with two corpora pointing at empty store paths."""
    # Neither store exists on disk, so create_app falls back to empty stores
    return AppConfig(
        provider=ProviderConfig(default="mock"),
        rag=RAGConfig(store_path=str(tmp_path / "store_default")),
        embedding=EmbeddingConfig(cache_dir=str(tmp_path / "emb_cache")),
        corpora={
            "fastapi": CorpusConfig(
                label="FastAPI Docs",
                store_path=str(tmp_path / "store_fastapi"),
                data_path="data/tech_docs",
                refusal_threshold=0.35,
            ),
            "k8s": CorpusConfig(
                label="Kubernetes",
                store_path=str(tmp_path / "store_k8s"),
                data_path="data/k8s_docs",
                refusal_threshold=0.30,
            ),
        },
        default_corpus="fastapi",
    )


def test_corpus_map_keys_match_config(multi_corpus_config):
    """app.state.corpus_map is keyed by corpus names."""
    app = create_app(multi_corpus_config)
    assert set(app.state.corpus_map.keys()) == {"fastapi", "k8s"}


def test_corpus_map_inner_dict_keyed_by_provider(multi_corpus_config):
    """Each corpus entry is a dict keyed by provider name (nested composition)."""
    app = create_app(multi_corpus_config)
    # Mock provider is the only one registered (no API keys set)
    for corpus_name in ("fastapi", "k8s"):
        inner = app.state.corpus_map[corpus_name]
        assert isinstance(inner, dict)
        assert "mock" in inner
        # Every inner dict has the same provider keys
        assert set(inner.keys()) == set(app.state.corpus_map["fastapi"].keys())


def test_default_orchestrator_points_at_default_corpus_and_provider(multi_corpus_config):
    """app.state.orchestrator == corpus_map[default_corpus][default_provider]."""
    app = create_app(multi_corpus_config)
    assert (
        app.state.orchestrator
        is app.state.corpus_map["fastapi"]["mock"]
    )


def test_legacy_mode_has_empty_corpus_map():
    """If config.corpora is empty, corpus_map is empty too."""
    config = AppConfig(provider=ProviderConfig(default="mock"))
    app = create_app(config)
    assert app.state.corpus_map == {}
    # Legacy orchestrator still attached
    assert app.state.orchestrator is not None


def test_default_corpus_not_in_corpora_raises():
    """Pydantic validator rejects default_corpus not in corpora."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="default_corpus"):
        AppConfig(
            corpora={
                "fastapi": CorpusConfig(
                    label="FastAPI Docs",
                    store_path=".cache/store",
                    data_path="data/tech_docs",
                ),
            },
            default_corpus="kubernetes",  # typo — should be "fastapi"
        )


def test_legacy_rag_refusal_threshold_preserved_when_no_corpora(tmp_path):
    """In legacy mode, rag.refusal_threshold drives the SearchTool."""
    from agent_bench.core.config import RAGConfig

    config = AppConfig(
        provider=ProviderConfig(default="mock"),
        rag=RAGConfig(
            store_path=str(tmp_path / "store"),
            refusal_threshold=0.42,
        ),
        embedding=EmbeddingConfig(cache_dir=str(tmp_path / "emb")),
    )
    app = create_app(config)
    # No corpora → empty corpus_map → legacy store attached
    assert app.state.corpus_map == {}
    # Legacy orchestrator's registry has the SearchTool built with the
    # legacy refusal_threshold (we reach into the tool registry to verify).
    search_tool = app.state.orchestrator.registry.get("search_documents")
    assert search_tool is not None
    assert search_tool.refusal_threshold == 0.42


def test_only_one_store_built_per_corpus(multi_corpus_config, monkeypatch):
    """In multi-corpus mode, the legacy single-store path is skipped.

    Counts HybridStore constructions: should equal len(config.corpora), not
    len(config.corpora) + 1 (the +1 being the now-deleted legacy store).
    """
    from agent_bench.rag import store as store_mod

    constructed: list = []
    orig_init = store_mod.HybridStore.__init__

    def tracking_init(self, *args, **kwargs):
        constructed.append(self)
        return orig_init(self, *args, **kwargs)

    monkeypatch.setattr(store_mod.HybridStore, "__init__", tracking_init)
    create_app(multi_corpus_config)
    # Exactly 2 stores (one per corpus). The legacy store is not built.
    assert len(constructed) == len(multi_corpus_config.corpora)


def test_corpus_map_has_all_providers(multi_corpus_config, monkeypatch):
    """With two providers available, each corpus inner dict has both.

    Verifies the structural invariant that every corpus exposes the same
    set of provider keys — the contract that Task 3's routing depends on.
    """
    from agent_bench.core import provider as provider_mod
    from agent_bench.core.provider import MockProvider

    class FakeOpenAI(MockProvider):
        pass

    monkeypatch.setattr(provider_mod, "OpenAIProvider", lambda _cfg: FakeOpenAI())
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    app = create_app(multi_corpus_config)
    expected_providers = {"mock", "openai"}
    for corpus_name in ("fastapi", "k8s"):
        inner = app.state.corpus_map[corpus_name]
        assert set(inner.keys()) == expected_providers
    # Structural invariant: every corpus has identical provider keys
    key_sets = [set(v.keys()) for v in app.state.corpus_map.values()]
    assert all(ks == key_sets[0] for ks in key_sets)
    # Provider orchestrators within a corpus are distinct instances
    assert (
        app.state.corpus_map["fastapi"]["mock"]
        is not app.state.corpus_map["fastapi"]["openai"]
    )
    # Same provider across corpora is also distinct (different registries)
    assert (
        app.state.corpus_map["fastapi"]["mock"]
        is not app.state.corpus_map["k8s"]["mock"]
    )


def test_unavailable_corpus_is_skipped(tmp_path):
    """A corpus with available=False is kept in config.corpora for
    schema visibility but is NOT wired into corpus_map at startup."""
    config = AppConfig(
        provider=ProviderConfig(default="mock"),
        rag=RAGConfig(store_path=str(tmp_path / "store_default")),
        embedding=EmbeddingConfig(cache_dir=str(tmp_path / "emb_cache")),
        corpora={
            "fastapi": CorpusConfig(
                label="FastAPI",
                store_path=str(tmp_path / "store_fastapi"),
                data_path="data/tech_docs",
            ),
            "k8s": CorpusConfig(
                label="Kubernetes",
                store_path=str(tmp_path / "store_k8s"),
                data_path="data/k8s_docs",
                available=False,
            ),
        },
        default_corpus="fastapi",
    )
    app = create_app(config)
    # Only fastapi wired in corpus_map
    assert set(app.state.corpus_map.keys()) == {"fastapi"}
    # But k8s is still in config.corpora for dashboard/introspection
    assert "k8s" in config.corpora
    assert config.corpora["k8s"].available is False


@pytest.mark.asyncio
async def test_unavailable_k8s_corpus_returns_400_at_request_time(tmp_path):
    """End-to-end: request for the unavailable corpus gets 400."""
    from httpx import ASGITransport, AsyncClient

    config = AppConfig(
        provider=ProviderConfig(default="mock"),
        rag=RAGConfig(store_path=str(tmp_path / "store_default")),
        embedding=EmbeddingConfig(cache_dir=str(tmp_path / "emb_cache")),
        corpora={
            "fastapi": CorpusConfig(
                label="FastAPI",
                store_path=str(tmp_path / "store_fastapi"),
                data_path="data/tech_docs",
            ),
            "k8s": CorpusConfig(
                label="Kubernetes",
                store_path=str(tmp_path / "store_k8s"),
                data_path="data/k8s_docs",
                available=False,
            ),
        },
        default_corpus="fastapi",
    )
    app = create_app(config)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/ask", json={"question": "hi", "corpus": "k8s"},
        )
    assert resp.status_code == 400
    assert "k8s" in resp.json()["detail"]
