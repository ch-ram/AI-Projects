"""
Production API Gateway for GenAI Platform.
Routes: Auth, RAG, Agents, Health, Metrics.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.auth.security import (
    User,
    authenticate_user,
    create_access_token,
    get_current_user,
    rate_limiter,
    require_scope,
    TokenResponse,
)
from src.monitoring.observability import setup_metrics, setup_tracing
from src.utils.config import get_settings
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# ── Global Services ─────────────────────────────────────────

rag_service = None
agent_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service, agent_service
    setup_logging()
    logger.info("platform_starting")

    try:
        from src.rag.service import RAGService
        rag_service = RAGService()
    except Exception as e:
        logger.warning("rag_service_init_failed", error=str(e))

    try:
        from src.agents.service import AgentService
        agent_service = AgentService()
    except Exception as e:
        logger.warning("agent_service_init_failed", error=str(e))

    yield
    logger.info("platform_shutting_down")


# ── App Setup ───────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title="GenAI Platform API",
    description="Production GenAI System — RAG, Agents, Auth, Monitoring",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_metrics(app)
setup_tracing(app)


# ── Request/Response Models ──────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = None
    use_rag: bool = True
    use_agent: bool = False
    system_prompt: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[list[dict]] = None
    intent: Optional[str] = None
    session_id: str
    model: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    services: dict


# ── Middleware ──────────────────────────────────────────────

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()

    response = await call_next(request)

    latency = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = f"{latency:.1f}"

    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round(latency, 1),
        request_id=request_id,
    )
    return response


# ── Auth Endpoints ─────────────────────────────────────────

@app.post("/api/v1/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(request: LoginRequest):
    """Authenticate and receive JWT token."""
    user = authenticate_user(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(user)
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expiration_minutes * 60,
    )


@app.get("/api/v1/auth/me", tags=["Auth"])
async def get_me(user: User = Depends(get_current_user)):
    """Get current authenticated user info."""
    return {"user_id": user.user_id, "email": user.email, "role": user.role}


# ── Query Endpoint (RAG + Agents) ──────────────────────────

@app.post("/api/v1/query", response_model=QueryResponse, tags=["Query"])
async def query(
    request: QueryRequest,
    user: User = Depends(get_current_user),
):
    """Unified query endpoint with RAG and/or Agent support."""
    # Rate limiting
    if not rate_limiter.check(user.user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    session_id = request.session_id or str(uuid4())[:8]
    start = time.perf_counter()

    try:
        answer = ""
        sources = None
        intent = None
        model = settings.primary_model

        if request.use_rag and rag_service:
            result = await rag_service.query(request.question)
            answer = result["answer"]
            sources = result["sources"]

        elif request.use_agent and agent_service:
            result = await agent_service.execute(
                request.question,
                system_prompt=request.system_prompt,
            )
            answer = result["result"]
            intent = result["intent"]
            model = result["model"]

        else:
            # Direct LLM call
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=settings.primary_model)
            response = await llm.ainvoke(request.question)
            answer = response.content

        latency = (time.perf_counter() - start) * 1000

        return QueryResponse(
            answer=answer,
            sources=sources,
            intent=intent,
            session_id=session_id,
            model=model,
            latency_ms=round(latency, 1),
        )

    except Exception as e:
        logger.error("query_failed", error=str(e), user=user.user_id)
        raise HTTPException(status_code=500, detail=str(e))


# ── Admin Endpoints ────────────────────────────────────────

@app.post(
    "/api/v1/admin/ingest",
    tags=["Admin"],
    dependencies=[Depends(require_scope("admin"))],
)
async def ingest_documents(directory: str):
    """Ingest documents into RAG (admin only)."""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not available")

    from src.rag.service import Document
    from pathlib import Path

    # Load documents from directory
    docs = []
    p = Path(directory)
    if p.is_dir():
        for f in p.rglob("*.txt"):
            docs.append(Document(
                page_content=f.read_text(),
                metadata={"source": str(f)},
            ))

    count = await rag_service.ingest(docs)
    return {"status": "success", "documents_ingested": count}


# ── Health & Info ──────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.app_env.value,
        services={
            "rag": rag_service is not None,
            "agents": agent_service is not None,
            "auth": True,
        },
    )


@app.get("/ready", tags=["Health"])
async def readiness():
    """Kubernetes readiness probe."""
    if rag_service is None and agent_service is None:
        raise HTTPException(status_code=503, detail="Services not ready")
    return {"status": "ready"}


@app.get("/live", tags=["Health"])
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


# ── Entry Point ─────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
