"""FastAPI application factory."""

from __future__ import annotations

import os
import time
from pathlib import Path

import psutil
import structlog
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
    log = structlog.get_logger()

    # Load task config for system prompt
    task = load_task_config("tech_docs")

    # Providers — create all available, keyed by name
    provider = create_provider(config)
    providers: dict = {config.provider.default: provider}
    _alt_providers = {"openai", "anthropic"} - {config.provider.default}
    for alt in _alt_providers:
        try:
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

    # --- Shared RAG components (corpus-independent) ---
    embedder = Embedder(
        model_name=config.embedding.model,
        cache_dir=config.embedding.cache_dir,
    )

    reranker = None
    if config.rag.reranker.enabled:
        from agent_bench.rag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=config.rag.reranker.model_name)

    # --- Security components (constructed before tools so PII redactor
    # can be injected into per-corpus SearchTools) ---
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
        secret_check=sec.output.secret_check,
        blocklist=sec.output.blocklist,
    )
    audit_logger = AuditLogger(
        path=sec.audit.path,
        max_size_bytes=sec.audit.max_size_mb * 1024 * 1024,
        rotate=sec.audit.rotate,
    )

    # --- Mode-dependent construction: multi-corpus vs legacy single-corpus ---
    corpus_map: dict[str, dict[str, Orchestrator]] = {}
    orchestrators: dict[str, Orchestrator] = {}
    store: HybridStore

    if config.corpora:
        # Multi-corpus mode. Skip the legacy single-store path entirely —
        # each corpus gets its own store / retriever / registry, and the
        # per-corpus inner dict holds one Orchestrator per available provider.
        _proc = psutil.Process()
        _baseline_rss = _proc.memory_info().rss / 1024**2

        _default_store: HybridStore | None = None

        for corpus_name, corpus_cfg in config.corpora.items():
            # Skip corpora marked unavailable. They stay in config.corpora
            # for schema visibility but are not wired into corpus_map,
            # so routes return 400 via _resolve_orchestrator and the
            # dashboard can render the toggle as disabled.
            if not corpus_cfg.available:
                log.warning(
                    "corpus_skipped_unavailable",
                    name=corpus_name,
                    label=corpus_cfg.label,
                    reason="CorpusConfig.available=False",
                    hint="set available=true once the store is built",
                )
                continue

            c_store_path = Path(corpus_cfg.store_path)
            if c_store_path.exists() and (c_store_path / "index.faiss").exists():
                c_store = HybridStore.load(
                    str(c_store_path), rrf_k=config.rag.retrieval.rrf_k,
                )
            else:
                c_store = HybridStore(
                    dimension=384, rrf_k=config.rag.retrieval.rrf_k,
                )

            c_retriever = Retriever(
                embedder=embedder,
                store=c_store,
                default_strategy=config.rag.retrieval.strategy,  # type: ignore[arg-type]
                candidates_per_system=config.rag.retrieval.candidates_per_system,
                reranker=reranker,
                reranker_top_k=config.rag.reranker.top_k,
            )
            c_registry = ToolRegistry()
            c_registry.register(
                SearchTool(
                    retriever=c_retriever,
                    default_top_k=corpus_cfg.top_k,
                    default_strategy=config.rag.retrieval.strategy,  # type: ignore[arg-type]
                    refusal_threshold=corpus_cfg.refusal_threshold,
                    pii_redactor=pii_redactor if sec.pii.enabled else None,
                )
            )
            c_registry.register(CalculatorTool())

            inner: dict[str, Orchestrator] = {}
            for p_name, p_prov in providers.items():
                inner[p_name] = Orchestrator(
                    provider=p_prov,
                    registry=c_registry,
                    max_iterations=corpus_cfg.max_iterations,
                    temperature=config.agent.temperature,
                )
            corpus_map[corpus_name] = inner

            if corpus_name == config.default_corpus:
                _default_store = c_store

            _rss_mb = _proc.memory_info().rss / 1024**2
            log.info(
                "corpus_loaded",
                name=corpus_name,
                label=corpus_cfg.label,
                store_path=str(c_store_path),
                providers=list(inner.keys()),
                rss_mb=round(_rss_mb, 1),
                rss_delta_mb=round(_rss_mb - _baseline_rss, 1),
            )

        log.info(
            "multi_corpus_mode",
            corpora=list(corpus_map.keys()),
            default=config.default_corpus,
            providers=list(providers.keys()),
        )

        # Legacy rag.refusal_threshold is ignored in multi-corpus mode;
        # per-corpus refusal_threshold is authoritative. Only warn when the
        # legacy value is non-default AND differs from the default corpus's
        # threshold — that is the actual drift case. A legacy value that
        # matches the default corpus is benign (someone kept both in sync).
        legacy_thresh = config.rag.refusal_threshold
        default_thresh = config.corpora[config.default_corpus].refusal_threshold
        if legacy_thresh != 0.0 and legacy_thresh != default_thresh:
            log.warning(
                "rag_refusal_threshold_drift_in_multi_corpus_mode",
                legacy_value=legacy_thresh,
                default_corpus=config.default_corpus,
                default_corpus_value=default_thresh,
                hint="rag.refusal_threshold is ignored; "
                     "update corpora.<name>.refusal_threshold instead",
            )

        # AppConfig._validate_default_corpus guarantees default_corpus is in
        # corpora when corpora is non-empty, so _default_store is always set.
        assert _default_store is not None
        store = _default_store
        # orchestrators (flat, per-provider) is the default-corpus inner dict
        # — keeps /ask's existing provider-switching code path working for
        # the default corpus. Per-request corpus routing in Task 3 will
        # consult corpus_map[corpus][provider] directly.
        orchestrators = dict(corpus_map[config.default_corpus])
        orchestrator = orchestrators[config.provider.default]
    else:
        # Legacy single-corpus mode.
        log.info("single_corpus_mode_legacy")

        store_path = Path(config.rag.store_path)
        if store_path.exists() and (store_path / "index.faiss").exists():
            store = HybridStore.load(str(store_path), rrf_k=config.rag.retrieval.rrf_k)
        else:
            store = HybridStore(dimension=384, rrf_k=config.rag.retrieval.rrf_k)

        retriever = Retriever(
            embedder=embedder,
            store=store,
            default_strategy=config.rag.retrieval.strategy,  # type: ignore[arg-type]
            candidates_per_system=config.rag.retrieval.candidates_per_system,
            reranker=reranker,
            reranker_top_k=config.rag.reranker.top_k,
        )

        registry = ToolRegistry()
        registry.register(
            SearchTool(
                retriever=retriever,
                default_top_k=config.rag.retrieval.top_k,
                default_strategy=config.rag.retrieval.strategy,  # type: ignore[arg-type]
                refusal_threshold=config.rag.refusal_threshold,
                pii_redactor=pii_redactor if sec.pii.enabled else None,
            )
        )
        registry.register(CalculatorTool())

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
    app.state.corpus_map = corpus_map
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
        log.info("warmup_start")
        _ = embedder.embed("warmup")
        if reranker is not None:
            _ = reranker.model  # noqa: F841
        log.info("warmup_complete")

    return app
