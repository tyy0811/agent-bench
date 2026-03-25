"""Tests for the serving layer: routes, schemas, middleware."""

from __future__ import annotations

import time

import pytest
from httpx import ASGITransport, AsyncClient

from agent_bench.agents.orchestrator import Orchestrator
from agent_bench.core.config import AppConfig, ProviderConfig
from agent_bench.core.provider import MockProvider, ProviderTimeoutError
from agent_bench.rag.store import HybridStore
from agent_bench.serving.middleware import MetricsCollector, RateLimitMiddleware, RequestMiddleware
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
        assert data["metadata"]["provider"] == "mock"
        assert data["metadata"]["model"] == "mock-1"

    @pytest.mark.asyncio
    async def test_empty_question_returns_422(self, test_app):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post("/ask", json={"question": ""})
        assert response.status_code == 422

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


# --- Rate limiting tests ---


def _make_rate_limited_app(rpm: int = 3):
    """Create a test app with rate limiting enabled."""
    from fastapi import FastAPI

    app = FastAPI(title="agent-bench-ratelimit")

    registry = ToolRegistry()
    registry.register(FakeSearchTool())
    registry.register(CalculatorTool())

    provider = MockProvider()
    orchestrator = Orchestrator(provider=provider, registry=registry, max_iterations=3)

    app.state.orchestrator = orchestrator
    app.state.store = HybridStore(dimension=384)
    app.state.config = AppConfig(provider=ProviderConfig(default="mock"))
    app.state.system_prompt = "You are a test assistant."
    app.state.start_time = time.time()
    app.state.metrics = MetricsCollector()

    app.add_middleware(RequestMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=rpm)

    from agent_bench.serving.routes import router

    app.include_router(router)
    return app


@pytest.fixture
def rate_limited_app():
    return _make_rate_limited_app(rpm=3)


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_allows_normal_traffic(self, rate_limited_app):
        """Requests within the limit all succeed."""
        async with AsyncClient(
            transport=ASGITransport(app=rate_limited_app), base_url="http://test"
        ) as client:
            for _ in range(3):
                response = await client.get("/health")
                assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_blocks_excess(self, rate_limited_app):
        """Request beyond the limit gets 429."""
        async with AsyncClient(
            transport=ASGITransport(app=rate_limited_app), base_url="http://test"
        ) as client:
            # Use up the quota
            for _ in range(3):
                await client.post("/ask", json={"question": "test"})
            # Next request should be blocked
            response = await client.post("/ask", json={"question": "test"})
            assert response.status_code == 429

    @pytest.mark.asyncio
    async def test_retry_after_header(self, rate_limited_app):
        """429 response includes Retry-After header."""
        async with AsyncClient(
            transport=ASGITransport(app=rate_limited_app), base_url="http://test"
        ) as client:
            # Exhaust quota on non-exempt path
            for _ in range(3):
                await client.post("/ask", json={"question": "test"})
            response = await client.post("/ask", json={"question": "test"})
            assert response.status_code == 429
            assert "retry-after" in response.headers
            assert int(response.headers["retry-after"]) > 0

    @pytest.mark.asyncio
    async def test_health_exempt(self):
        """Health endpoint is never rate limited."""
        app = _make_rate_limited_app(rpm=2)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Exhaust quota on non-exempt path
            for _ in range(2):
                await client.post("/ask", json={"question": "test"})
            # Health should still work
            response = await client.get("/health")
            assert response.status_code == 200
            # But another ask should be blocked
            response = await client.post("/ask", json={"question": "test"})
            assert response.status_code == 429
