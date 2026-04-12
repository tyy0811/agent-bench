"""Tests for corpus + corpus_label fields in the SSE meta event.

The multi-corpus fixture is auto-loaded from tests/conftest.py.
"""

from __future__ import annotations

import json as json_mod

import pytest
from httpx import ASGITransport, AsyncClient


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.strip().split("\n"):
        if line.startswith("data: "):
            events.append(json_mod.loads(line[6:]))
    return events


class TestMetaCorpus:
    @pytest.mark.asyncio
    async def test_meta_includes_corpus_and_label_default(
        self, two_corpus_two_provider_app,
    ):
        app = two_corpus_two_provider_app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post("/ask/stream", json={"question": "hi"})
        events = _parse_sse(resp.text)
        meta = next(e for e in events if e.get("type") == "meta")
        assert meta["metadata"]["corpus"] == "fastapi"
        assert meta["metadata"]["corpus_label"] == "FastAPI Docs"

    @pytest.mark.asyncio
    async def test_meta_reflects_explicit_corpus(
        self, two_corpus_two_provider_app,
    ):
        app = two_corpus_two_provider_app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/ask/stream", json={"question": "hi", "corpus": "k8s"},
            )
        events = _parse_sse(resp.text)
        meta = next(e for e in events if e.get("type") == "meta")
        assert meta["metadata"]["corpus"] == "k8s"
        assert meta["metadata"]["corpus_label"] == "Kubernetes"

    @pytest.mark.asyncio
    async def test_meta_reflects_resolved_provider_not_config_default(
        self, two_corpus_two_provider_app,
    ):
        """Meta event must report the actually-resolved provider, not
        config.provider.default. Adversarial review flagged that the
        previous implementation would say 'openai' in the meta event
        even when the request was routed to anthropic (or vice versa).
        """
        app = two_corpus_two_provider_app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/ask/stream",
                json={"question": "hi", "corpus": "k8s", "provider": "openai"},
            )
        events = _parse_sse(resp.text)
        meta = next(e for e in events if e.get("type") == "meta")
        assert meta["metadata"]["corpus"] == "k8s"
        assert meta["metadata"]["corpus_label"] == "Kubernetes"
        # Config default is 'mock', but the request asked for 'openai'
        # and openai IS wired — meta must say openai.
        assert meta["metadata"]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_meta_provider_matches_default_when_implicit(
        self, two_corpus_two_provider_app,
    ):
        """When body.provider is None, meta reports the config default."""
        app = two_corpus_two_provider_app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/ask/stream", json={"question": "hi"},
            )
        events = _parse_sse(resp.text)
        meta = next(e for e in events if e.get("type") == "meta")
        assert meta["metadata"]["provider"] == "mock"  # config default
