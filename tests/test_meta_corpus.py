"""Tests for corpus + corpus_label fields in the SSE meta event."""

from __future__ import annotations

import json as json_mod

import pytest
from httpx import ASGITransport, AsyncClient

from tests.test_corpus_routing import two_corpus_two_provider_app  # noqa: F401


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.strip().split("\n"):
        if line.startswith("data: "):
            events.append(json_mod.loads(line[6:]))
    return events


class TestMetaCorpus:
    @pytest.mark.asyncio
    async def test_meta_includes_corpus_and_label_default(
        self, two_corpus_two_provider_app,  # noqa: F811
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
        self, two_corpus_two_provider_app,  # noqa: F811
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
    async def test_meta_provider_label_composes_with_corpus(
        self, two_corpus_two_provider_app,  # noqa: F811
    ):
        """Provider field in meta still reflects the config default."""
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
        # Meta is emitted before the actual orchestrator runs; config.default
        # provider is what's advertised. Corpus metadata follows the request.
        assert meta["metadata"]["corpus"] == "k8s"
        assert meta["metadata"]["corpus_label"] == "Kubernetes"
