"""API routes: /ask, /ask/stream, /health, /metrics, /metrics/prometheus."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.responses import Response

from agent_bench.agents.orchestrator import Orchestrator
from agent_bench.core.config import AppConfig
from agent_bench.core.prompts import format_system_prompt
from agent_bench.serving.middleware import MetricsCollector
from agent_bench.serving.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    MetricsResponse,
    ResponseMetadata,
)

router = APIRouter()


def _resolve_orchestrator(
    request: Request, body: AskRequest,
) -> tuple[Orchestrator, str, str]:
    """Resolve (orchestrator, corpus_name, provider_name) for a request.

    Multi-corpus mode: look up corpus_map[corpus][provider]. If the
    request explicitly names a provider that isn't wired for the
    resolved corpus, raise 400 instead of silently falling back —
    silent fallback makes the provider comparison telemetry
    untrustworthy and hides config drift.

    Legacy single-corpus mode: use the flat orchestrators dict keyed by
    provider name. Same strict rule: explicit body.provider that isn't
    in orchestrators → 400. Implicit (None) → fall through to default.

    Raises:
        HTTPException(400): body.corpus names a corpus not in corpus_map,
            OR body.provider names a provider not wired for the resolved
            corpus. Pydantic Literal catches unknown names at 422; this
            catches "known per schema but not deployed at runtime" at 400.

    Returns:
        (orchestrator, corpus_name, provider_name). provider_name is
        the actual provider key used to reach the orchestrator — it
        may differ from body.provider when body.provider is None and
        the corpus default is used.
    """
    config: AppConfig = request.app.state.config
    corpus_map: dict = getattr(request.app.state, "corpus_map", {})
    default_corpus: str = getattr(config, "default_corpus", "") or ""
    provider_default: str = config.provider.default

    # Fail loud on unwired corpus.
    if corpus_map and body.corpus is not None and body.corpus not in corpus_map:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Corpus {body.corpus!r} is not configured on this server. "
                f"Available corpora: {sorted(corpus_map.keys())}"
            ),
        )

    corpus_name: str = body.corpus or default_corpus

    if corpus_map and corpus_name in corpus_map:
        inner = corpus_map[corpus_name]
        # Explicit body.provider must be wired for this corpus. No silent
        # fallback — we'd mislabel telemetry and lie in the meta event.
        if body.provider is not None:
            if body.provider not in inner:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Provider {body.provider!r} is not available for "
                        f"corpus {corpus_name!r}. Available providers: "
                        f"{sorted(inner.keys())}"
                    ),
                )
            return inner[body.provider], corpus_name, body.provider
        # Implicit — use the corpus's copy of the config default provider.
        # If even the default isn't wired (misconfig), 500 is appropriate;
        # we let KeyError propagate as a loud server error.
        return inner[provider_default], corpus_name, provider_default

    # Legacy single-corpus mode: flat per-provider dict.
    orchestrators: dict = getattr(request.app.state, "orchestrators", {})
    if body.provider is not None:
        if body.provider not in orchestrators:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Provider {body.provider!r} is not available. "
                    f"Available providers: {sorted(orchestrators.keys())}"
                ),
            )
        return orchestrators[body.provider], corpus_name, body.provider
    return request.app.state.orchestrator, corpus_name, provider_default


def _resolve_system_prompt(
    request: Request, corpus_name: str,
) -> tuple[str, str]:
    """Return (system_prompt, corpus_label) for the active corpus.

    In multi-corpus mode the prompt is formatted from the shared template
    with the corpus's label substituted in. In legacy mode, the prompt
    from the task config (app.state.system_prompt) is returned unchanged
    and corpus_label is empty.
    """
    config: AppConfig = request.app.state.config
    corpora = getattr(config, "corpora", None) or {}
    if corpus_name and corpus_name in corpora:
        label = corpora[corpus_name].label
        return format_system_prompt(label), label
    return request.app.state.system_prompt, ""


_LANDING_HTML_TEMPLATE: str | None = None


def _get_landing_html_template() -> str:
    """Read and cache the raw index.html template on first call."""
    global _LANDING_HTML_TEMPLATE  # noqa: PLW0603
    if _LANDING_HTML_TEMPLATE is None:
        from pathlib import Path

        html_path = Path(__file__).parent / "static" / "index.html"
        _LANDING_HTML_TEMPLATE = html_path.read_text()
    return _LANDING_HTML_TEMPLATE


def _render_landing_html(config: AppConfig) -> str:
    """Inject per-server corpus availability into the cached HTML.

    The dashboard reads the JSON from a <script id="corpus-config">
    block to decide which corpus toggles to enable. Injection uses a
    literal string replace rather than a template engine to keep the
    landing page a single static file.
    """
    import json as _json

    template = _get_landing_html_template()
    corpora_data = {
        name: {"label": cfg.label, "available": cfg.available}
        for name, cfg in config.corpora.items()
    }
    payload = _json.dumps({
        "corpora": corpora_data,
        "default_corpus": config.default_corpus,
    })
    # Escape </script> to avoid HTML injection if a config value ever
    # contains one. json.dumps already escapes backslashes and quotes.
    payload = payload.replace("</", "<\\/")
    return template.replace("{{CORPUS_CONFIG_JSON}}", payload)


@router.get("/")
async def root(request: Request) -> Response:
    """Showcase landing page with live RAG dashboard."""
    from starlette.responses import HTMLResponse

    return HTMLResponse(content=_render_landing_html(request.app.state.config))


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """Ask a question and get an answer with sources."""
    orchestrator, corpus_name, _provider_name = _resolve_orchestrator(request, body)
    system_prompt, _corpus_label = _resolve_system_prompt(request, corpus_name)
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
    orchestrator, corpus_name, provider_name = _resolve_orchestrator(request, body)
    system_prompt, corpus_label = _resolve_system_prompt(request, corpus_name)
    metrics: MetricsCollector = request.app.state.metrics
    request_id: str = getattr(request.state, "request_id", "unknown")
    config: AppConfig = request.app.state.config

    # --- Meta event data (resolved from the actual orchestrator, not
    # from config.provider.default — otherwise a dashboard request with
    # provider="anthropic" would see "openai" in the meta event).
    provider_obj = orchestrator.provider
    model_name = getattr(
        provider_obj, "model_name",
        getattr(provider_obj, "_model_name", provider_name),
    )

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
            "provider": provider_name,
            "model": model_name,
            "corpus": corpus_name,
            "corpus_label": corpus_label,
            "config": {
                "top_k": body.top_k,
                "max_iterations": (
                    config.agent.max_iterations
                    if getattr(config, "agent", None) else 3
                ),
                "strategy": body.retrieval_strategy,
            },
        }).to_sse()

        # --- Injection check stage ---
        yield StreamEvent(type="stage", metadata={
            "stage": "injection_check",
            "status": "done",
            "verdict": injection_verdict_data,
        }).to_sse()

        # Stream orchestrator events live. Stage events are yielded
        # immediately so the dashboard can animate in real time.
        # Only the chunk content is accumulated for post-stream
        # output validation (monitor mode).
        full_answer: list[str] = []
        done_meta: dict = {}
        async for event in orchestrator.run_stream(
            question=body.question,
            system_prompt=system_prompt,
            top_k=body.top_k,
            strategy=body.retrieval_strategy,
            history=history,
        ):
            if event.type == "_orchestrator_done":
                # Extract metadata, don't yield to client
                if event.metadata:
                    done_meta = event.metadata
                continue
            if event.type == "chunk" and event.content:
                full_answer.append(event.content)
                # Don't yield chunk yet — validate first
                continue
            # Yield stage and sources events live
            yield event.to_sse()

        # --- Security: output validation (post-generation, monitor mode) ---
        answer_text = "".join(full_answer)
        filtered_answer = answer_text
        output_verdict_data: dict = {"passed": True, "violations": []}
        output_blocked = False
        source_chunks = done_meta.get("source_chunks", [])
        if output_validator:
            out_verdict = output_validator.validate(
                output=answer_text,
                retrieved_chunks=source_chunks,
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

        # Yield the (possibly filtered) answer chunk
        yield StreamEvent(
            type="chunk",
            content=filtered_answer if output_blocked else answer_text,
        ).to_sse()

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
            "pii_redactions_count": done_meta.get(
                "pii_redactions_count", 0,
            ),
        }).to_sse()

        # Record metrics and persist session
        cost = done_meta.get("estimated_cost_usd", 0.0)
        metrics.record(latency_ms=latency_ms, cost_usd=cost)

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
