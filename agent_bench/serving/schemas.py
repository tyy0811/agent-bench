"""Request and response Pydantic models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent_bench.agents.orchestrator import SourceReference
from agent_bench.core.types import TokenUsage


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    retrieval_strategy: Literal["semantic", "keyword", "hybrid"] = "hybrid"


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
