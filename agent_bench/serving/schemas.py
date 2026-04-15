"""Request and response Pydantic models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent_bench.agents.orchestrator import SourceReference
from agent_bench.core.types import TokenUsage


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = 5
    retrieval_strategy: Literal["semantic", "keyword", "hybrid"] = "hybrid"
    session_id: str | None = None  # None = stateless (V1 behavior)
    # Per-request provider override. Constrained to the set of known
    # provider names so unknown values are rejected at validation time
    # with HTTP 422 instead of silently falling back.
    provider: Literal["openai", "anthropic", "selfhosted", "mock"] | None = None
    # Per-request corpus selection. None = use default_corpus from config.
    # Unknown values rejected at validation time with HTTP 422. Names that
    # pass validation but are not wired on the current server produce a
    # 400 in the route handler (see _resolve_orchestrator).
    corpus: Literal["fastapi", "k8s"] | None = None


class ResponseMetadata(BaseModel):
    provider: str
    model: str
    iterations: int
    tools_used: list[str]
    latency_ms: float
    token_usage: TokenUsage
    request_id: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    metadata: ResponseMetadata


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    vector_store_chunks: int
    provider_available: bool
    uptime_seconds: float


class MetricsResponse(BaseModel):
    requests_total: int
    latency_p50_ms: float
    latency_p95_ms: float
    errors_total: int
    avg_cost_per_query_usd: float


class StreamEvent(BaseModel):
    """SSE event for streaming responses."""

    type: str  # "sources", "chunk", "done"
    content: str | None = None
    sources: list[dict] | None = None
    metadata: dict | None = None

    def to_sse(self) -> str:
        return f"data: {self.model_dump_json()}\n\n"
