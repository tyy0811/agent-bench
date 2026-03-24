"""API routes: /ask, /health, /metrics."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from agent_bench.agents.orchestrator import Orchestrator
from agent_bench.serving.middleware import MetricsCollector
from agent_bench.serving.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    MetricsResponse,
    ResponseMetadata,
)

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """Ask a question and get an answer with sources."""
    orchestrator: Orchestrator = request.app.state.orchestrator
    system_prompt: str = request.app.state.system_prompt
    metrics: MetricsCollector = request.app.state.metrics
    request_id: str = getattr(request.state, "request_id", "unknown")

    result = await orchestrator.run(
        question=body.question,
        system_prompt=system_prompt,
        top_k=body.top_k,
        strategy=body.retrieval_strategy,
    )

    metrics.record(
        latency_ms=result.latency_ms,
        cost_usd=result.usage.estimated_cost_usd,
    )

    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        metadata=ResponseMetadata(
            provider=result.provider,
            model=result.model,
            iterations=result.iterations,
            tools_used=result.tools_used,
            latency_ms=result.latency_ms,
            token_usage=result.usage,
            request_id=request_id,
        ),
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    store = request.app.state.store
    start_time: float = request.app.state.start_time

    provider_available = True
    try:
        # Just check the provider is constructed — don't make an API call
        _ = request.app.state.orchestrator.provider
    except Exception:
        provider_available = False

    return HealthResponse(
        status="healthy" if provider_available else "degraded",
        vector_store_chunks=store.stats().total_chunks,
        provider_available=provider_available,
        uptime_seconds=time.time() - start_time,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(request: Request) -> MetricsResponse:
    """Metrics endpoint."""
    m: MetricsCollector = request.app.state.metrics

    return MetricsResponse(
        requests_total=m.requests_total,
        latency_p50_ms=m.percentile(50),
        latency_p95_ms=m.percentile(95),
        errors_total=m.errors_total,
        avg_cost_per_query_usd=m.avg_cost,
    )
