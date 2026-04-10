"""Tests for route-level SSE events: meta, injection_check, output_validation."""

import json as json_mod
import time

import pytest
from httpx import ASGITransport, AsyncClient

from agent_bench.agents.orchestrator import Orchestrator
from agent_bench.core.config import AppConfig, ProviderConfig, SecurityConfig
from agent_bench.core.provider import MockProvider
from agent_bench.rag.store import HybridStore
from agent_bench.serving.middleware import MetricsCollector, RequestMiddleware
from agent_bench.tools.calculator import CalculatorTool
from agent_bench.tools.registry import ToolRegistry
from tests.test_agent import FakeSearchTool


def _parse_sse(response_text):
    events = []
    for line in response_text.strip().split("\n"):
        if line.startswith("data: "):
            events.append(json_mod.loads(line[6:]))
    return events


def _make_app_with_security(tmp_path):
    from fastapi import FastAPI

    from agent_bench.security.audit_logger import AuditLogger
    from agent_bench.security.injection_detector import InjectionDetector
    from agent_bench.security.output_validator import OutputValidator
    from agent_bench.security.pii_redactor import PIIRedactor

    config = AppConfig(
        provider=ProviderConfig(default="mock"),
        security=SecurityConfig(),
    )
    config.security.audit.path = str(tmp_path / "audit.jsonl")

    app = FastAPI()
    registry = ToolRegistry()
    registry.register(FakeSearchTool())
    registry.register(CalculatorTool())

    provider = MockProvider()
    orchestrator = Orchestrator(provider=provider, registry=registry, max_iterations=3)

    app.state.orchestrator = orchestrator
    app.state.store = HybridStore(dimension=384)
    app.state.config = config
    app.state.system_prompt = "You are a test assistant."
    app.state.start_time = time.time()
    app.state.metrics = MetricsCollector()
    app.state.injection_detector = InjectionDetector(tiers=["heuristic"], enabled=True)
    app.state.pii_redactor = PIIRedactor(mode="redact")
    app.state.output_validator = OutputValidator()
    app.state.audit_logger = AuditLogger(path=str(tmp_path / "audit.jsonl"))

    app.add_middleware(RequestMiddleware)
    from agent_bench.serving.routes import router
    app.include_router(router)
    return app


class TestMetaEvent:
    @pytest.mark.asyncio
    async def test_first_event_is_meta(self, tmp_path):
        app = _make_app_with_security(tmp_path)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/ask/stream", json={"question": "How do path params work?"})

        events = _parse_sse(resp.text)
        assert events[0]["type"] == "meta"
        assert "provider" in events[0]["metadata"]
        assert "model" in events[0]["metadata"]

    @pytest.mark.asyncio
    async def test_meta_includes_config(self, tmp_path):
        app = _make_app_with_security(tmp_path)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/ask/stream", json={"question": "test"})

        events = _parse_sse(resp.text)
        meta = events[0]["metadata"]
        assert "config" in meta
        assert "top_k" in meta["config"]
        assert "max_iterations" in meta["config"]


class TestInjectionStageEvent:
    @pytest.mark.asyncio
    async def test_injection_check_stage_emitted(self, tmp_path):
        app = _make_app_with_security(tmp_path)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/ask/stream", json={"question": "How do path params work?"})

        events = _parse_sse(resp.text)
        stage_events = [e for e in events if e["type"] == "stage"]
        injection_done = [e for e in stage_events
                          if e["metadata"].get("stage") == "injection_check"
                          and e["metadata"].get("status") == "done"]
        assert len(injection_done) == 1
        assert injection_done[0]["metadata"]["verdict"]["safe"] is True


class TestOutputValidationStageEvent:
    @pytest.mark.asyncio
    async def test_output_validation_after_chunk(self, tmp_path):
        app = _make_app_with_security(tmp_path)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/ask/stream", json={"question": "How do path params work?"})

        events = _parse_sse(resp.text)
        types = [e["type"] for e in events]

        # output_validation stage must come after chunk
        chunk_idx = next(i for i, t in enumerate(types) if t == "chunk")
        ov_indices = [i for i, e in enumerate(events)
                      if e["type"] == "stage"
                      and e.get("metadata", {}).get("stage") == "output_validation"]
        assert len(ov_indices) == 1
        assert ov_indices[0] > chunk_idx

    @pytest.mark.asyncio
    async def test_output_validation_mode_is_monitor(self, tmp_path):
        app = _make_app_with_security(tmp_path)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/ask/stream", json={"question": "test"})

        events = _parse_sse(resp.text)
        ov = [e for e in events if e["type"] == "stage"
              and e.get("metadata", {}).get("stage") == "output_validation"]
        assert ov[0]["metadata"]["mode"] == "monitor"


class TestDoneEventEnriched:
    @pytest.mark.asyncio
    async def test_done_has_latency_and_tokens(self, tmp_path):
        app = _make_app_with_security(tmp_path)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/ask/stream", json={"question": "test"})

        events = _parse_sse(resp.text)
        done = [e for e in events if e["type"] == "done"][0]
        meta = done["metadata"]
        assert "latency_ms" in meta
        assert "tokens_in" in meta
        assert "tokens_out" in meta
        assert "iterations" in meta


class TestFullEventSequence:
    @pytest.mark.asyncio
    async def test_complete_event_ordering(self, tmp_path):
        """Full sequence: meta -> injection -> stages -> sources -> chunk -> output_val -> done."""
        app = _make_app_with_security(tmp_path)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/ask/stream", json={"question": "How do path params work?"})

        events = _parse_sse(resp.text)
        types = [(e["type"], (e.get("metadata") or {}).get("stage")) for e in events]

        # First event is meta
        assert types[0] == ("meta", None)

        # Second is injection_check
        assert types[1] == ("stage", "injection_check")

        # Last two: output_validation stage then done
        assert types[-2] == ("stage", "output_validation")
        assert types[-1][0] == "done"

        # sources and chunk exist somewhere in the middle
        flat_types = [t[0] for t in types]
        assert "sources" in flat_types
        assert "chunk" in flat_types
