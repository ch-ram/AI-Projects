"""
Production observability stack.
OpenTelemetry tracing + Prometheus metrics + structured logging.
"""
from __future__ import annotations

import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    Counter,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Prometheus Metrics ─────────────────────────────────────

APP_INFO = Info("genai_platform", "GenAI Platform information")

REQUEST_COUNT = Counter(
    "genai_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "genai_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

LLM_REQUEST_COUNT = Counter(
    "genai_llm_requests_total",
    "Total LLM API calls",
    ["model", "endpoint_type"],
)

LLM_LATENCY = Histogram(
    "genai_llm_request_duration_seconds",
    "LLM request latency",
    ["model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

LLM_TOKENS = Counter(
    "genai_llm_tokens_total",
    "Total LLM tokens used",
    ["model", "type"],  # type: prompt | completion
)

RAG_RETRIEVAL_LATENCY = Histogram(
    "genai_rag_retrieval_seconds",
    "RAG retrieval latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

ACTIVE_SESSIONS = Counter(
    "genai_active_sessions_total",
    "Total active user sessions",
)


# ── Metrics Middleware ─────────────────────────────────────

class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect HTTP request metrics for Prometheus."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start
        endpoint = request.url.path
        method = request.method
        status = str(response.status_code)

        REQUEST_COUNT.labels(
            method=method, endpoint=endpoint, status_code=status
        ).inc()
        REQUEST_LATENCY.labels(
            method=method, endpoint=endpoint
        ).observe(duration)

        return response


# ── OpenTelemetry Setup ────────────────────────────────────

def setup_tracing(app: FastAPI) -> None:
    """Initialize OpenTelemetry tracing."""
    settings = get_settings()

    if not settings.enable_tracing:
        logger.info("tracing_disabled")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({
            "service.name": settings.otel_service_name,
            "service.version": settings.app_version,
            "deployment.environment": settings.app_env.value,
        })

        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)

        logger.info("tracing_initialized", endpoint=settings.otel_endpoint)

    except ImportError:
        logger.warning("opentelemetry_not_installed")
    except Exception as e:
        logger.error("tracing_setup_failed", error=str(e))


# ── Metrics Endpoint ───────────────────────────────────────

def setup_metrics(app: FastAPI) -> None:
    """Add Prometheus metrics endpoint and middleware."""
    settings = get_settings()

    APP_INFO.info({
        "version": settings.app_version,
        "environment": settings.app_env.value,
    })

    app.add_middleware(MetricsMiddleware)

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return StarletteResponse(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    logger.info("metrics_endpoint_registered")
