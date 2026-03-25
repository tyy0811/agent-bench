"""API routes: /ask, /ask/stream, /health, /metrics."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from starlette.responses import Response

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


@router.get("/")
async def root() -> Response:
    """Human-friendly landing page for recruiters clicking the live URL."""
    from starlette.responses import HTMLResponse

    html = (  # noqa: E501
        "<!DOCTYPE html>"
        "<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>agent-bench</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:640px;"
        "margin:60px auto;padding:0 20px;color:#1a1a1a;line-height:1.6}"
        "h1{margin-bottom:4px}.sub{color:#666;margin-top:0}"
        "code{background:#f4f4f4;padding:2px 6px;border-radius:3px}"
        "pre{background:#f4f4f4;padding:16px;border-radius:6px;"
        "overflow-x:auto}a{color:#0066cc}"
        "table{border-collapse:collapse;width:100%;margin:12px 0}"
        "th,td{text-align:left;padding:8px 12px;"
        "border-bottom:1px solid #e0e0e0}th{font-weight:600}"
        "</style></head><body>"
        "<h1>agent-bench</h1>"
        "<p class='sub'>RAG agent evaluation benchmark"
        " &mdash; built from API primitives</p>"
        "<table>"
        "<tr><th>Endpoint</th><th>Description</th></tr>"
        "<tr><td><code>POST /ask</code></td>"
        "<td>Ask a question, get answer with sources</td></tr>"
        "<tr><td><code>POST /ask/stream</code></td>"
        "<td>SSE streaming</td></tr>"
        "<tr><td><code>GET /health</code></td>"
        "<td>Health check and store stats</td></tr>"
        "<tr><td><code>GET /metrics</code></td>"
        "<td>Request count, latency, cost</td></tr>"
        "</table>"
        "<h3>Try it</h3>"
        "<pre>curl -X POST "
        "https://nomearod-agentbench.hf.space/ask \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -d '{\"question\": "
        "\"How do I add auth to FastAPI?\"}'</pre>"
        "<p><strong>145 tests</strong> &middot; "
        "<strong>2 providers</strong> (OpenAI + Anthropic)"
        " &middot; <strong>27-question benchmark</strong></p>"
        "<p><a href='https://github.com/tyy0811/agent-bench'>"
        "GitHub</a></p>"
        "</body></html>"
    )
    return HTMLResponse(content=html)


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """Ask a question and get an answer with sources."""
    orchestrator: Orchestrator = request.app.state.orchestrator
    system_prompt: str = request.app.state.system_prompt
    metrics: MetricsCollector = request.app.state.metrics
    request_id: str = getattr(request.state, "request_id", "unknown")

    # Load conversation history if session_id provided
    history: list[dict] | None = None
    conversation_store = getattr(request.app.state, "conversation_store", None)
    if body.session_id and conversation_store:
        max_turns = request.app.state.config.memory.max_turns
        history = conversation_store.get_history(body.session_id, max_turns=max_turns)

    result = await orchestrator.run(
        question=body.question,
        system_prompt=system_prompt,
        top_k=body.top_k,
        strategy=body.retrieval_strategy,
        history=history,
    )

    # Store Q+A if session_id provided
    if body.session_id and conversation_store:
        conversation_store.append(body.session_id, "user", body.question)
        conversation_store.append(body.session_id, "assistant", result.answer)

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


@router.post("/ask/stream")
async def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
    """Stream an answer via Server-Sent Events."""
    orchestrator: Orchestrator = request.app.state.orchestrator
    system_prompt: str = request.app.state.system_prompt

    async def event_generator():
        async for event in orchestrator.run_stream(
            question=body.question,
            system_prompt=system_prompt,
            top_k=body.top_k,
            strategy=body.retrieval_strategy,
        ):
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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
