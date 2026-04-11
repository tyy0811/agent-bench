"""FastAPI application factory."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI

from agent_bench.agents.orchestrator import Orchestrator
from agent_bench.core.config import AppConfig, load_config, load_task_config
from agent_bench.core.provider import create_provider
from agent_bench.rag.embedder import Embedder
from agent_bench.rag.retriever import Retriever
from agent_bench.rag.store import HybridStore
from agent_bench.serving.middleware import MetricsCollector, RateLimitMiddleware, RequestMiddleware
from agent_bench.serving.routes import router
from agent_bench.tools.calculator import CalculatorTool
from agent_bench.tools.registry import ToolRegistry
from agent_bench.tools.search import SearchTool


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Initializes all singletons and attaches them to app.state.
    """
    if config is None:
        config = load_config()

    app = FastAPI(title="agent-bench", version="0.1.0")

    # Load task config for system prompt
    task = load_task_config("tech_docs")

    # Providers — create all available, keyed by name
    provider = create_provider(config)
    providers: dict = {config.provider.default: provider}
    _alt_providers = {"openai", "anthropic"} - {config.provider.default}
    for alt in _alt_providers:
        try:
            import os

            from agent_bench.core.provider import (
                AnthropicProvider,
                OpenAIProvider,
            )

            if alt == "openai" and os.environ.get("OPENAI_API_KEY"):
                providers["openai"] = OpenAIProvider(config)
            elif alt == "anthropic" and os.environ.get(
                "ANTHROPIC_API_KEY",
            ):
                providers["anthropic"] = AnthropicProvider(config)
        except Exception:
            pass  # missing dependency or key — skip

    # RAG pipeline
    store_path = Path(config.rag.store_path)
    if store_path.exists() and (store_path / "index.faiss").exists():
        store = HybridStore.load(str(store_path), rrf_k=config.rag.retrieval.rrf_k)
        embedder = Embedder(
            model_name=config.embedding.model,
            cache_dir=config.embedding.cache_dir,
        )
    else:
        # No store on disk — create empty store (for testing or first run)
        store = HybridStore(dimension=384, rrf_k=config.rag.retrieval.rrf_k)
        embedder = Embedder(
            model_name=config.embedding.model,
            cache_dir=config.embedding.cache_dir,
        )

    # Optional reranker
    reranker = None
    if config.rag.reranker.enabled:
        from agent_bench.rag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=config.rag.reranker.model_name)

    retriever = Retriever(
        embedder=embedder,
        store=store,
        default_strategy=config.rag.retrieval.strategy,  # type: ignore[arg-type]
        candidates_per_system=config.rag.retrieval.candidates_per_system,
        reranker=reranker,
        reranker_top_k=config.rag.reranker.top_k,
    )

    # Security components (constructed before tools so PII redactor can be injected)
    from agent_bench.security.audit_logger import AuditLogger
    from agent_bench.security.injection_detector import InjectionDetector
    from agent_bench.security.output_validator import OutputValidator
    from agent_bench.security.pii_redactor import PIIRedactor

    sec = config.security
    injection_detector = InjectionDetector(
        tiers=sec.injection.tiers,
        classifier_url=sec.injection.classifier_url,
        enabled=sec.injection.enabled,
    )
    pii_redactor = PIIRedactor(
        redact_patterns=sec.pii.redact_patterns,
        mode=sec.pii.mode,
        use_ner=sec.pii.use_ner,
    )
    output_validator = OutputValidator(
        pii_check=sec.output.pii_check,
        url_check=sec.output.url_check,
        blocklist=sec.output.blocklist,
    )
    audit_logger = AuditLogger(
        path=sec.audit.path,
        max_size_bytes=sec.audit.max_size_mb * 1024 * 1024,
        rotate=sec.audit.rotate,
    )

    # Tools (PII redactor injected into search tool for post-retrieval redaction)
    registry = ToolRegistry()
    registry.register(
        SearchTool(
            retriever=retriever,
            default_top_k=config.rag.retrieval.top_k,
            default_strategy=config.rag.retrieval.strategy,
            refusal_threshold=config.rag.refusal_threshold,
            pii_redactor=pii_redactor if sec.pii.enabled else None,
        )
    )
    registry.register(CalculatorTool())

    # Orchestrators — one per available provider
    orchestrators: dict = {}
    for name, prov in providers.items():
        orchestrators[name] = Orchestrator(
            provider=prov,
            registry=registry,
            max_iterations=config.agent.max_iterations,
            temperature=config.agent.temperature,
        )
    orchestrator = orchestrators[config.provider.default]

    # Metrics
    metrics = MetricsCollector()

    # Conversation memory (optional, SQLite-backed)
    conversation_store = None
    if config.memory.enabled:
        from agent_bench.memory.store import ConversationStore

        conversation_store = ConversationStore(db_path=config.memory.db_path)

    # Attach to app state
    app.state.orchestrator = orchestrator
    app.state.orchestrators = orchestrators
    app.state.store = store
    app.state.conversation_store = conversation_store
    app.state.config = config
    app.state.system_prompt = task.system_prompt
    app.state.start_time = time.time()
    app.state.metrics = metrics
    app.state.injection_detector = injection_detector
    app.state.pii_redactor = pii_redactor
    app.state.output_validator = output_validator
    app.state.audit_logger = audit_logger

    # Middleware and routes (order matters: rate limit checked first)
    app.add_middleware(RequestMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=config.serving.rate_limit_rpm)
    app.include_router(router)

    # Startup warmup: eager-load models to reduce cold start latency
    @app.on_event("startup")
    async def warmup() -> None:
        import structlog

        log = structlog.get_logger()
        log.info("warmup_start")
        _ = embedder.embed("warmup")
        if reranker is not None:
            _ = reranker.model  # noqa: F841
        log.info("warmup_complete")

    return app
