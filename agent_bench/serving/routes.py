"""API routes: /ask, /ask/stream, /health, /metrics, /metrics/prometheus."""

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
        "<p><strong>169 tests</strong> &middot; "
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

    # --- Security: injection detection (pre-retrieval) ---
    injection_detector = getattr(request.app.state, "injection_detector", None)
    injection_verdict_data = {"safe": True, "tier": "none", "confidence": 1.0}
    if injection_detector:
        verdict = await injection_detector.detect_async(body.question)
        injection_verdict_data = {
            "safe": verdict.safe,
            "tier": verdict.tier,
            "confidence": verdict.confidence,
            "matched_pattern": verdict.matched_pattern,
        }
        sec_config = getattr(request.app.state.config, "security", None)
        action = sec_config.injection.action if sec_config else "block"
        if not verdict.safe and action == "block":
            # Log blocked request to audit
            _write_audit(request, body, request_id, injection_verdict_data, blocked=True)
            from fastapi.responses import JSONResponse
            return JSONResponse(  # type: ignore[return-value]
                status_code=403,
                content={
                    "detail": "Request blocked: potential prompt injection detected",
                    "request_id": request_id,
                },
            )

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

    # --- Security: output validation (post-generation) ---
    output_verdict_data: dict = {"passed": True, "violations": []}
    output_validator = getattr(request.app.state, "output_validator", None)
    answer = result.answer
    if output_validator:
        out_verdict = output_validator.validate(
            output=result.answer,
            retrieved_chunks=result.source_chunks,
        )
        output_verdict_data = {
            "passed": out_verdict.passed,
            "violations": out_verdict.violations,
        }
        if not out_verdict.passed and out_verdict.action == "block":
            answer = (
                "I'm unable to provide a response to this query. "
                "The output was filtered for safety."
            )

    # Store Q+A if session_id provided
    if body.session_id and conversation_store:
        conversation_store.append(body.session_id, "user", body.question)
        conversation_store.append(body.session_id, "assistant", answer)

    metrics.record(
        latency_ms=result.latency_ms,
        cost_usd=result.usage.estimated_cost_usd,
    )

    response = AskResponse(
        answer=answer,
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

    # --- Security: audit log ---
    _write_audit(
        request, body, request_id, injection_verdict_data,
        result=result, output_verdict_data=output_verdict_data,
    )

    return response


@router.post("/ask/stream")
async def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
    """Stream an answer via Server-Sent Events with per-stage instrumentation."""
    orchestrator: Orchestrator = request.app.state.orchestrator
    system_prompt: str = request.app.state.system_prompt
    metrics: MetricsCollector = request.app.state.metrics
    request_id: str = getattr(request.state, "request_id", "unknown")
    config: object = request.app.state.config

    # --- Meta event data (available before request starts) ---
    provider_name = getattr(config, "provider", None)
    provider_default = getattr(provider_name, "default", "unknown") if provider_name else "unknown"
    provider_obj = orchestrator.provider
    model_name = getattr(provider_obj, "model_name", getattr(provider_obj, "_model_name", provider_default))

    # --- Security: injection detection (pre-retrieval) ---
    injection_detector = getattr(request.app.state, "injection_detector", None)
    injection_verdict_data = {"safe": True, "tier": "none", "confidence": 1.0}
    if injection_detector:
        verdict = await injection_detector.detect_async(body.question)
        injection_verdict_data = {
            "safe": verdict.safe,
            "tier": verdict.tier,
            "confidence": verdict.confidence,
            "matched_pattern": verdict.matched_pattern,
        }
        sec_config = getattr(request.app.state.config, "security", None)
        action = sec_config.injection.action if sec_config else "block"
        if not verdict.safe and action == "block":
            _write_audit(
                request, body, request_id, injection_verdict_data,
                endpoint="/ask/stream", blocked=True,
            )
            from fastapi.responses import JSONResponse
            return JSONResponse(  # type: ignore[return-value]
                status_code=403,
                content={
                    "detail": "Request blocked: potential prompt injection detected",
                    "request_id": request_id,
                },
            )

    # Load conversation history if session_id provided
    history: list[dict] | None = None
    conversation_store = getattr(request.app.state, "conversation_store", None)
    if body.session_id and conversation_store:
        max_turns = request.app.state.config.memory.max_turns
        history = conversation_store.get_history(body.session_id, max_turns=max_turns)

    start = time.perf_counter()
    output_validator = getattr(request.app.state, "output_validator", None)

    async def event_generator():
        from agent_bench.serving.schemas import StreamEvent

        # --- Meta event (first, before any stages) ---
        yield StreamEvent(type="meta", metadata={
            "provider": provider_default,
            "model": model_name,
            "config": {
                "top_k": body.top_k,
                "max_iterations": getattr(config, "agent", None) and config.agent.max_iterations or 3,
                "strategy": body.retrieval_strategy,
            },
        }).to_sse()

        # --- Injection check stage ---
        yield StreamEvent(type="stage", metadata={
            "stage": "injection_check",
            "status": "done",
            "verdict": injection_verdict_data,
        }).to_sse()

        # Buffer orchestrator events for output validation
        buffered_events: list = []
        full_answer: list[str] = []
        done_meta: dict = {}
        async for event in orchestrator.run_stream(
            question=body.question,
            system_prompt=system_prompt,
            top_k=body.top_k,
            strategy=body.retrieval_strategy,
            history=history,
        ):
            buffered_events.append(event)
            if event.type == "chunk" and event.content:
                full_answer.append(event.content)
            if event.type == "_orchestrator_done" and event.metadata:
                done_meta = event.metadata

        # --- Security: output validation (post-generation, monitor mode) ---
        answer_text = "".join(full_answer)
        filtered_answer = answer_text
        output_verdict_data: dict = {"passed": True, "violations": []}
        output_blocked = False
        if output_validator:
            out_verdict = output_validator.validate(
                output=answer_text,
                retrieved_chunks=[],
            )
            output_verdict_data = {
                "passed": out_verdict.passed,
                "violations": out_verdict.violations,
            }
            if not out_verdict.passed and out_verdict.action == "block":
                output_blocked = True
                filtered_answer = (
                    "I'm unable to provide a response to this query. "
                    "The output was filtered for safety."
                )

        # Yield buffered orchestrator events (stage events + legacy events)
        # Filter out _orchestrator_done — route handler emits the real done event
        for event in buffered_events:
            if event.type == "_orchestrator_done":
                continue
            if output_blocked and event.type == "chunk":
                yield StreamEvent(type="chunk", content=filtered_answer).to_sse()
            else:
                yield event.to_sse()

        # --- Output validation stage (monitor mode, after chunk) ---
        yield StreamEvent(type="stage", metadata={
            "stage": "output_validation",
            "status": "done",
            "mode": "monitor",
            "verdict": {
                "passed": output_verdict_data["passed"],
                "violations": output_verdict_data.get("violations", []),
            },
        }).to_sse()

        # --- Enriched done event with latency ---
        latency_ms = (time.perf_counter() - start) * 1000
        yield StreamEvent(type="done", metadata={
            "latency_ms": latency_ms,
            "tokens_in": done_meta.get("tokens_in", 0),
            "tokens_out": done_meta.get("tokens_out", 0),
            "cost": done_meta.get("estimated_cost_usd", 0.0),
            "iterations": done_meta.get("iterations", 1),
        }).to_sse()

        # Record metrics and persist session
        metrics.record(latency_ms=latency_ms, cost_usd=done_meta.get("estimated_cost_usd", 0.0))

        if body.session_id and conversation_store:
            conversation_store.append(body.session_id, "user", body.question)
            conversation_store.append(body.session_id, "assistant", filtered_answer)

        # Audit log
        _write_audit(
            request, body, request_id, injection_verdict_data,
            endpoint="/ask/stream",
            output_verdict_data=output_verdict_data,
        )

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

    provider_available = False
    try:
        provider = request.app.state.orchestrator.provider
        provider_available = await provider.health_check()
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


@router.get("/metrics/prometheus")
async def metrics_prometheus(request: Request) -> Response:
    """Prometheus text exposition format for K8s HPA custom metrics."""
    m: MetricsCollector = request.app.state.metrics
    lines = [
        "# HELP agent_bench_requests_total Total requests served.",
        "# TYPE agent_bench_requests_total counter",
        f"agent_bench_requests_total {m.requests_total}",
        "# HELP agent_bench_errors_total Total error responses.",
        "# TYPE agent_bench_errors_total counter",
        f"agent_bench_errors_total {m.errors_total}",
        "# HELP agent_bench_latency_p50_ms 50th percentile latency in ms.",
        "# TYPE agent_bench_latency_p50_ms gauge",
        f"agent_bench_latency_p50_ms {m.percentile(50):.1f}",
        "# HELP agent_bench_latency_p95_ms 95th percentile latency in ms.",
        "# TYPE agent_bench_latency_p95_ms gauge",
        f"agent_bench_latency_p95_ms {m.percentile(95):.1f}",
        "# HELP agent_bench_avg_cost_usd Average cost per query in USD.",
        "# TYPE agent_bench_avg_cost_usd gauge",
        f"agent_bench_avg_cost_usd {m.avg_cost:.6f}",
        "",
    ]
    return Response(
        content="\n".join(lines),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


def _write_audit(
    request: Request,
    body: AskRequest,
    request_id: str,
    injection_verdict: dict,
    endpoint: str = "/ask",
    blocked: bool = False,
    result: object | None = None,
    output_verdict_data: dict | None = None,
) -> None:
    """Write an audit record if audit logger is configured."""
    audit_logger = getattr(request.app.state, "audit_logger", None)
    if not audit_logger:
        return

    client_ip = request.client.host if request.client else "unknown"

    record: dict = {
        "request_id": request_id,
        "session_id": body.session_id,
        "client_ip": audit_logger.hash_ip(client_ip),
        "endpoint": endpoint,
        "input_query": body.question,
        "injection_verdict": injection_verdict,
    }

    if blocked:
        record["blocked"] = True
    else:
        if result is not None:
            record.update({
                "retrieved_chunks": [s.source for s in getattr(result, "sources", [])],
                "llm_provider": getattr(result, "provider", ""),
                "llm_model": getattr(result, "model", ""),
                "output_tokens": getattr(getattr(result, "usage", None), "output_tokens", None),
                "grounded_refusal": not bool(getattr(result, "sources", [])),
                "response_latency_ms": getattr(result, "latency_ms", 0),
            })
        if output_verdict_data is not None:
            record["output_validation"] = output_verdict_data

    audit_logger.log(record)
