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
