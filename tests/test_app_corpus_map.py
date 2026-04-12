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
