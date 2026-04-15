"""Tests for the parameterized system prompt template.

The integration tests rely on `two_corpus_two_provider_app` from
tests/conftest.py.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from agent_bench.core.prompts import SYSTEM_PROMPT_TEMPLATE, format_system_prompt


def test_template_has_placeholder():
    assert "{corpus_label}" in SYSTEM_PROMPT_TEMPLATE


def test_format_substitutes_label():
    out = format_system_prompt("Kubernetes")
    assert "Kubernetes" in out
    assert "{corpus_label}" not in out


def test_format_distinct_labels_produce_distinct_prompts():
    a = format_system_prompt("FastAPI Docs")
    b = format_system_prompt("Kubernetes")
    assert a != b
    assert "FastAPI Docs" in a
    assert "Kubernetes" in b


def test_format_refusal_language():
    """Template uses 'refuse explicitly', not soft 'say so'."""
    out = format_system_prompt("FastAPI Docs")
    assert "refuse" in out.lower()


def test_format_prohibits_inference():
    """Template prohibits inference / extrapolation / general knowledge."""
    out = format_system_prompt("FastAPI Docs")
    text = out.lower()
    assert "do not infer" in text
    assert "extrapolate" in text
    assert "general knowledge" in text


def test_format_requires_citations():
    """Template still requires source citations in [source: file.md] form."""
    out = format_system_prompt("FastAPI Docs")
    assert "[source:" in out


def test_format_rejects_empty_label():
    """Empty label is a caller bug — fail loud instead of producing a
    prompt with an unresolved placeholder."""
    with pytest.raises(ValueError, match="corpus_label"):
        format_system_prompt("")


def test_format_is_cached():
    """@lru_cache on format_system_prompt — same input returns same object."""
    a = format_system_prompt("FastAPI Docs")
    b = format_system_prompt("FastAPI Docs")
    assert a is b  # cached: same object identity, not just equal


class TestRouteHandlerUsesFormattedPrompt:
    """In multi-corpus mode the orchestrator must receive a prompt
    formatted with the active corpus's label — not the legacy
    app.state.system_prompt."""

    @pytest.mark.asyncio
    async def test_stream_passes_k8s_prompt_to_orchestrator(
        self, two_corpus_two_provider_app,
    ):
        app = two_corpus_two_provider_app
        # Record every system_prompt the orchestrator sees.
        captured: list[str] = []
        target_orch = app.state.corpus_map["k8s"]["mock"]
        orig_run_stream = target_orch.run_stream

        async def spy_run_stream(*args, **kwargs):
            captured.append(kwargs.get("system_prompt", ""))
            async for event in orig_run_stream(*args, **kwargs):
                yield event

        target_orch.run_stream = spy_run_stream  # type: ignore[method-assign]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/ask/stream", json={"question": "hi", "corpus": "k8s"},
            )
        assert resp.status_code == 200
        assert len(captured) == 1
        prompt = captured[0]
        # Prompt must be the formatted multi-corpus template, not the
        # legacy app.state.system_prompt.
        assert "Kubernetes" in prompt
        assert "{corpus_label}" not in prompt
        assert "refuse" in prompt.lower()

    @pytest.mark.asyncio
    async def test_fastapi_and_k8s_prompts_differ(
        self, two_corpus_two_provider_app,
    ):
        app = two_corpus_two_provider_app
        captured: dict[str, str] = {}

        def _make_spy(corpus_name: str, orch):
            orig = orch.run_stream

            async def spy(*args, **kwargs):
                captured[corpus_name] = kwargs.get("system_prompt", "")
                async for event in orig(*args, **kwargs):
                    yield event
            return spy

        fa = app.state.corpus_map["fastapi"]["mock"]
        ks = app.state.corpus_map["k8s"]["mock"]
        fa.run_stream = _make_spy("fastapi", fa)  # type: ignore[method-assign]
        ks.run_stream = _make_spy("k8s", ks)  # type: ignore[method-assign]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            await client.post(
                "/ask/stream", json={"question": "hi", "corpus": "fastapi"},
            )
            await client.post(
                "/ask/stream", json={"question": "hi", "corpus": "k8s"},
            )

        assert "FastAPI Docs" in captured["fastapi"]
        assert "Kubernetes" in captured["k8s"]
        assert captured["fastapi"] != captured["k8s"]
