"""Integration: the real landing render fills every reveal placeholder, exposes
the anchor JSON block, and the static DOM carries the revealed truth with NO
JS-only controls (no-JS visitors must see the truth and zero dead controls)."""
from __future__ import annotations

import json
import re

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
        },
        default_corpus="fastapi",
    )


def test_no_reveal_placeholder_survives(tmp_path):
    html = _render_landing_html(_make_config(tmp_path))
    assert "{{REVEAL_" not in html


def test_committed_values_present(tmp_path):
    html = _render_landing_html(_make_config(tmp_path))
    for value in [
        "0.791", "0.760", "+0.031", "[-0.013, +0.076]",
        "$0.0046", "$0.0007", "6.6x", "1.00", "0.14",
    ]:
        assert value in html, f"missing {value!r}"


def test_anchor_block_parseable(tmp_path):
    html = _render_landing_html(_make_config(tmp_path))
    match = re.search(
        r'<script id="reveal-anchor" type="application/json">(.*?)</script>',
        html, re.DOTALL,
    )
    assert match is not None, "reveal-anchor block missing"
    data = json.loads(match.group(1).replace("<\\/", "</"))
    assert data["collapse"]["a"]["p_at_5"] == 0.791
    assert data["cost"]["ratio"] == "6.6x"


def test_static_dom_is_revealed_truth_with_no_dead_controls(tmp_path):
    html = _render_landing_html(_make_config(tmp_path))
    # The self-sufficient revealed caption is present.
    assert "within noise" in html
    # The JS-only controls and resting caption are NOT in the static markup.
    assert "Show the confidence intervals" not in html
    assert "Real difference, or noise" not in html
    # No winner-asserting language in the reveal block itself.
    reveal_block = html.split("<!-- Reveal:")[1].split("<!-- Demo -->")[0]
    assert "ahead" not in reveal_block.lower()


@pytest.mark.asyncio
async def test_endpoint_serves_reveal(tmp_path):
    app = create_app(_make_config(tmp_path))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "{{REVEAL_" not in resp.text
    assert "6.6x" in resp.text
