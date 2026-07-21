"""The footer legal pages (/impressum, /privacy) serve their static HTML.

The IMPRESSUM_* placeholder tokens are pinned on purpose: they stay in the
page until Jane fills the legal details by hand, and the pin ensures nobody
ships an invented name or address by accident.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

STATIC_DIR = Path(__file__).parent.parent / "agent_bench" / "serving" / "static"


class TestStaticPages:
    @pytest.mark.asyncio
    async def test_impressum_serves_with_placeholder_tokens(
        self, two_corpus_two_provider_app,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=two_corpus_two_provider_app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/impressum")
        assert resp.status_code == 200
        for token in ("IMPRESSUM_NAME", "IMPRESSUM_ADDRESS", "IMPRESSUM_CONTACT"):
            assert token in resp.text

    @pytest.mark.asyncio
    async def test_privacy_serves_with_request_log_note(
        self, two_corpus_two_provider_app,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=two_corpus_two_provider_app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/privacy")
        assert resp.status_code == 200
        assert "request log" in resp.text
        assert "/impressum" in resp.text

    def test_footer_links_both_pages(self):
        html = (STATIC_DIR / "index.html").read_text()
        assert 'href="/impressum"' in html
        assert 'href="/privacy"' in html
