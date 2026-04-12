"""Tests for multi-corpus config schema."""

from agent_bench.core.config import AppConfig, CorpusConfig


def test_corpus_config_minimal_fields():
    c = CorpusConfig(
        label="FastAPI Docs",
        store_path=".cache/store",
        data_path="data/tech_docs",
    )
    assert c.label == "FastAPI Docs"
    assert c.refusal_threshold == 0.0  # default
    assert c.top_k == 5
    assert c.max_iterations == 3


def test_app_config_with_corpora():
    config = AppConfig.model_validate({
        "default_corpus": "fastapi",
        "corpora": {
            "fastapi": {
                "label": "FastAPI Docs",
                "store_path": ".cache/store",
                "data_path": "data/tech_docs",
                "refusal_threshold": 0.35,
                "top_k": 5,
                "max_iterations": 3,
            },
            "k8s": {
                "label": "Kubernetes",
                "store_path": ".cache/store_k8s",
                "data_path": "data/k8s_docs",
                "refusal_threshold": 0.30,
            },
        },
    })
    assert config.default_corpus == "fastapi"
    assert len(config.corpora) == 2
    assert config.corpora["k8s"].label == "Kubernetes"
    assert config.corpora["k8s"].refusal_threshold == 0.30


def test_app_config_empty_corpora_defaults():
    """Empty corpora dict is valid (legacy mode)."""
    config = AppConfig()
    assert config.corpora == {}
    assert config.default_corpus == "fastapi"


def test_corpus_config_available_defaults_true():
    """Back-compat: existing corpora without explicit 'available' are
    still wired. The flag defaults True."""
    c = CorpusConfig(
        label="FastAPI",
        store_path=".cache/store",
        data_path="data/tech_docs",
    )
    assert c.available is True


def test_corpus_config_available_can_be_false():
    c = CorpusConfig(
        label="K8s",
        store_path=".cache/store_k8s",
        data_path="data/k8s_docs",
        available=False,
    )
    assert c.available is False


def test_default_corpus_cannot_be_unavailable():
    """AppConfig validator rejects default_corpus pointing at an
    unavailable corpus — otherwise the app would boot with no
    reachable default orchestrator."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="available=False"):
        AppConfig.model_validate({
            "default_corpus": "fastapi",
            "corpora": {
                "fastapi": {
                    "label": "FastAPI",
                    "store_path": ".cache/store",
                    "data_path": "data/tech_docs",
                    "available": False,
                },
            },
        })


def test_non_default_corpus_can_be_unavailable():
    """A non-default unavailable corpus is valid — it's in the schema
    but will be skipped at startup."""
    config = AppConfig.model_validate({
        "default_corpus": "fastapi",
        "corpora": {
            "fastapi": {
                "label": "FastAPI",
                "store_path": ".cache/store",
                "data_path": "data/tech_docs",
            },
            "k8s": {
                "label": "K8s",
                "store_path": ".cache/store_k8s",
                "data_path": "data/k8s_docs",
                "available": False,
            },
        },
    })
    assert config.corpora["fastapi"].available is True
    assert config.corpora["k8s"].available is False
