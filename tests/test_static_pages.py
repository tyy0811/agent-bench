"""The footer legal pages (/impressum, /privacy) serve their static HTML.

Route coverage only. The IMPRESSUM_* and BOOKING_URL placeholder tokens are
deliberately NOT pinned here: CI must stay green both while the tokens are in
place and after Jane fills the real legal details. What must never happen is a
deploy with tokens unresolved, and that is the job of
scripts/preflight_deploy.py (make preflight-deploy), run before any push to
the hf remote.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

STATIC_DIR = Path(__file__).parent.parent / "agent_bench" / "serving" / "static"


def test_privacy_notice_is_concise_with_expandable_api_details():
    html = (STATIC_DIR / "privacy.html").read_text()
    text = " ".join(html.split())

    assert "This demo runs only in your browser and clears on reload." in text
    assert (
        "Direct API requests may be processed by an external AI provider and logged "
        "for security and debugging, so please do not submit personal data."
    ) in text
    assert "<details>" in html
    assert "<summary>API data handling</summary>" in html
    assert "session ID" in text
    assert "external providers process submitted content under their own terms" in text


class TestStaticPages:
    @pytest.mark.asyncio
    async def test_impressum_serves(self, two_corpus_two_provider_app):
        async with AsyncClient(
            transport=ASGITransport(app=two_corpus_two_provider_app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/impressum")
        assert resp.status_code == 200
        assert "<h1>Impressum</h1>" in resp.text

    @pytest.mark.asyncio
    async def test_privacy_serves(self, two_corpus_two_provider_app):
        async with AsyncClient(
            transport=ASGITransport(app=two_corpus_two_provider_app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/privacy")
        assert resp.status_code == 200
        assert "<h1>Privacy note</h1>" in resp.text
        assert "/impressum" in resp.text

    def test_footer_links_both_pages(self):
        html = (STATIC_DIR / "index.html").read_text()
        assert 'href="/impressum"' in html
        assert 'href="/privacy"' in html
