"""Tests that the landing-page HTML renderer injects per-server corpus
availability into the dashboard, so the dashboard can disable toggles
for unavailable corpora instead of silently 400'ing on click."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from agent_bench.core.config import (
    AppConfig,
    CorpusConfig,
    EmbeddingConfig,
    ProviderConfig,
    RAGConfig,
)
from agent_bench.serving.app import create_app
from agent_bench.serving.routes import _render_landing_html


def _make_config(tmp_path):
    return AppConfig(
        provider=ProviderConfig(default="mock"),
        rag=RAGConfig(store_path=str(tmp_path / "store_default")),
        embedding=EmbeddingConfig(cache_dir=str(tmp_path / "emb_cache")),
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
                available=False,
            ),
        },
        default_corpus="fastapi",
    )


def _extract_corpus_config_json(html: str) -> dict:
    """Pull the JSON out of <script id="corpus-config">...</script>."""
    import re

    match = re.search(
        r'<script id="corpus-config" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None, "corpus-config script block missing"
    # Reverse the </ -> <\/ escape we applied in _render_landing_html
    payload = match.group(1).replace("<\\/", "</")
    return json.loads(payload)


def test_placeholder_is_substituted(tmp_path):
    """The {{CORPUS_CONFIG_JSON}} placeholder must not survive rendering."""
    html = _render_landing_html(_make_config(tmp_path))
    assert "{{CORPUS_CONFIG_JSON}}" not in html


def test_injected_json_lists_all_corpora_with_availability(tmp_path):
    html = _render_landing_html(_make_config(tmp_path))
    data = _extract_corpus_config_json(html)
    assert data["default_corpus"] == "fastapi"
    assert set(data["corpora"].keys()) == {"fastapi", "k8s"}
    assert data["corpora"]["fastapi"]["available"] is True
    assert data["corpora"]["fastapi"]["label"] == "FastAPI Docs"
    assert data["corpora"]["k8s"]["available"] is False
    assert data["corpora"]["k8s"]["label"] == "Kubernetes"


@pytest.mark.asyncio
async def test_landing_page_endpoint_serves_rendered_html(tmp_path):
    """GET / returns the rendered HTML with corpus config injected."""
    config = _make_config(tmp_path)
    app = create_app(config)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "{{CORPUS_CONFIG_JSON}}" not in resp.text
    data = _extract_corpus_config_json(resp.text)
    assert data["corpora"]["k8s"]["available"] is False


def test_html_injection_escape(tmp_path):
    """If a corpus label ever contained </script>, the injected JSON
    must still be HTML-safe. json.dumps escapes quotes and backslashes;
    we additionally replace </ with <\\/ before emission."""
    config = AppConfig(
        provider=ProviderConfig(default="mock"),
        rag=RAGConfig(store_path=str(tmp_path / "store_default")),
        embedding=EmbeddingConfig(cache_dir=str(tmp_path / "emb_cache")),
        corpora={
            "fastapi": CorpusConfig(
                label="FastAPI </script><script>alert(1)</script>",
                store_path=str(tmp_path / "store_fastapi"),
                data_path="data/tech_docs",
            ),
        },
        default_corpus="fastapi",
    )
    html = _render_landing_html(config)
    # The evil script tag must not survive as a valid closer/opener
    # inside the <script id="corpus-config"> block.
    assert "</script><script>alert(1)</script>" not in html
    # But the label still round-trips through the JSON parse (with
    # the escape reversed) as the intended string.
    data = _extract_corpus_config_json(html)
    assert data["corpora"]["fastapi"]["label"] == (
        "FastAPI </script><script>alert(1)</script>"
    )
