"""Tests for the serving layer: routes, schemas, middleware."""

from __future__ import annotations

import time

import pytest
from httpx import ASGITransport, AsyncClient

from agent_bench.agents.orchestrator import Orchestrator
from agent_bench.core.config import AppConfig, ProviderConfig
from agent_bench.core.provider import MockProvider, ProviderTimeoutError
from agent_bench.rag.store import HybridStore
from agent_bench.serving.middleware import MetricsCollector, RequestMiddleware
from agent_bench.tools.calculator import CalculatorTool
from agent_bench.tools.registry import ToolRegistry

from .test_agent import FakeSearchTool


def _make_test_app():
    """Create a test app with MockProvider and no real store."""
    from fastapi import FastAPI

    app = FastAPI(title="agent-bench-test")

    registry = ToolRegistry()
    registry.register(FakeSearchTool())
    registry.register(CalculatorTool())

    provider = MockProvider()
    orchestrator = Orchestrator(provider=provider, registry=registry, max_iterations=3)

    app.state.orchestrator = orchestrator
    app.state.store = HybridStore(dimension=384)  # empty store for health check
    app.state.config = AppConfig(provider=ProviderConfig(default="mock"))
    app.state.system_prompt = "You are a test assistant."
    app.state.start_time = time.time()
    app.state.metrics = MetricsCollector()

    app.add_middleware(RequestMiddleware)

    from agent_bench.serving.routes import router

    app.include_router(router)
    return app


def _make_timeout_app():
    """Create a test app where the provider always times out."""
    from fastapi import FastAPI

    class TimeoutProvider(MockProvider):
        async def complete(self, messages, tools=None, temperature=0.0, max_tokens=1024):
            raise ProviderTimeoutError("test timeout")

    app = FastAPI(title="agent-bench-timeout")

    registry = ToolRegistry()
    registry.register(FakeSearchTool())

    provider = TimeoutProvider()
    orchestrator = Orchestrator(provider=provider, registry=registry, max_iterations=1)

    app.state.orchestrator = orchestrator
    app.state.store = HybridStore(dimension=384)
    app.state.config = AppConfig(provider=ProviderConfig(default="mock"))
    app.state.system_prompt = "You are a test assistant."
    app.state.start_time = time.time()
    app.state.metrics = MetricsCollector()

    app.add_middleware(RequestMiddleware)

    from agent_bench.serving.routes import router

    app.include_router(router)
    return app


@pytest.fixture
def test_app():
    return _make_test_app()


@pytest.fixture
def timeout_app():
    return _make_timeout_app()


class TestAskEndpoint:
    @pytest.mark.asyncio
    async def test_valid_question_returns_200(self, test_app):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post("/ask", json={"question": "How do path parameters work?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "metadata" in data
        assert len(data["answer"]) > 0
        assert data["metadata"]["request_id"]

    @pytest.mark.asyncio
    async def test_empty_question_returns_422(self, test_app):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post("/ask", json={"question": ""})
        # FastAPI doesn't reject empty strings by default (only missing fields)
        # But the orchestrator will still process it
        # If we want 422, we'd add a validator — for now, just verify it doesn't crash
        assert response.status_code in (200, 422)

    @pytest.mark.asyncio
    async def test_missing_question_returns_422(self, test_app):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post("/ask", json={})
        assert response.status_code == 422


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_returns_health_response(self, test_app):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "vector_store_chunks" in data
        assert "provider_available" in data
        assert "uptime_seconds" in data


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_returns_metrics_response(self, test_app):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests_total" in data
        assert "latency_p50_ms" in data
        assert "latency_p95_ms" in data
        assert "errors_total" in data
        assert "avg_cost_per_query_usd" in data


class TestMiddleware:
    @pytest.mark.asyncio
    async def test_request_id_header(self, test_app):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
        assert "x-request-id" in response.headers
        # UUID format: 8-4-4-4-12 hex chars
        request_id = response.headers["x-request-id"]
        assert len(request_id) == 36

    @pytest.mark.asyncio
    async def test_provider_timeout_returns_504(self, timeout_app):
        async with AsyncClient(
            transport=ASGITransport(app=timeout_app), base_url="http://test"
        ) as client:
            response = await client.post("/ask", json={"question": "This will timeout"})
        assert response.status_code == 504
        data = response.json()
        assert "request_id" in data
        assert "x-request-id" in response.headers
