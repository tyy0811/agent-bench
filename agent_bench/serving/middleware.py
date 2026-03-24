"""Request middleware: ID generation, logging, error handling, metrics."""

from __future__ import annotations

import time
import uuid
from collections import deque

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from agent_bench.core.provider import ProviderTimeoutError

logger = structlog.get_logger()


class MetricsCollector:
    """In-process metrics. Resets on restart."""

    def __init__(self, maxlen: int = 1000) -> None:
        self.latencies: deque[float] = deque(maxlen=maxlen)
        self.requests_total: int = 0
        self.errors_total: int = 0
        self.total_cost_usd: float = 0.0

    def record(self, latency_ms: float, cost_usd: float = 0.0, error: bool = False) -> None:
        self.latencies.append(latency_ms)
        self.requests_total += 1
        self.total_cost_usd += cost_usd
        if error:
            self.errors_total += 1

    def percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * p / 100)
        idx = min(idx, len(sorted_latencies) - 1)
        return sorted_latencies[idx]

    @property
    def avg_cost(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.total_cost_usd / self.requests_total


class RequestMiddleware(BaseHTTPMiddleware):
    """Adds request ID, timing, structured logging, and error handling."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        try:
            response = await call_next(request)
            latency_ms = (time.perf_counter() - start) * 1000

            response.headers["X-Request-ID"] = request_id

            logger.info(
                "request_completed",
                method=request.method,
                path=str(request.url.path),
                status=response.status_code,
                latency_ms=round(latency_ms, 2),
                request_id=request_id,
            )

            return response

        except ProviderTimeoutError:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "provider_timeout",
                method=request.method,
                path=str(request.url.path),
                latency_ms=round(latency_ms, 2),
                request_id=request_id,
            )
            # Record error in metrics if available
            metrics = getattr(request.app.state, "metrics", None)
            if metrics is not None:
                metrics.record(latency_ms, error=True)
            return JSONResponse(
                status_code=504,
                content={"detail": "Provider timed out", "request_id": request_id},
                headers={"X-Request-ID": request_id},
            )

        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "unhandled_error",
                method=request.method,
                path=str(request.url.path),
                latency_ms=round(latency_ms, 2),
                request_id=request_id,
            )
            metrics = getattr(request.app.state, "metrics", None)
            if metrics is not None:
                metrics.record(latency_ms, error=True)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
                headers={"X-Request-ID": request_id},
            )
