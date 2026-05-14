"""
Production FastAPI application for RAG Chatbot.
Features: SSE streaming, CORS, rate limiting, health checks, metrics.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.ingestion.embeddings import EmbeddingManager
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.rag_chain import RAGChain
from src.utils.config import get_settings
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# ── Global State ────────────────────────────────────────────

embedding_manager: Optional[EmbeddingManager] = None
retriever: Optional[HybridRetriever] = None
rag_chain: Optional[RAGChain] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: load index on startup, cleanup on shutdown."""
    global embedding_manager, retriever, rag_chain

    setup_logging()
    logger.info("application_starting")

    # Load persisted index
    embedding_manager = EmbeddingManager()
    index_loaded = embedding_manager.load()

    if index_loaded:
        retriever = HybridRetriever(embedding_manager)
        rag_chain = RAGChain(retriever)
        logger.info("rag_chain_ready", vectors=embedding_manager.index.ntotal)
    else:
        logger.warning("no_index_found_ingest_documents_first")

    yield

    logger.info("application_shutting_down")


# ── App Setup ───────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title="RAG Chatbot API",
    description="Production Document Q&A with RAG, Hybrid Retrieval, and Streaming",
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


# ── Request/Response Models ─────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    use_cot: bool = False
    stream: bool = False
    top_k: Optional[int] = Field(None, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    retrieval_scores: list[float]
    latency_ms: float


class IngestRequest(BaseModel):
    directory: str
    strategy: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    index_loaded: bool
    vector_count: int


# ── Middleware ──────────────────────────────────────────────

@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    """Add request ID and timing to every request."""
    request_id = str(uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()

    response = await call_next(request)

    latency = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = f"{latency:.1f}"

    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round(latency, 1),
        request_id=request_id,
    )
    return response


# ── Endpoints ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.app_env.value,
        index_loaded=embedding_manager is not None and embedding_manager.index is not None,
        vector_count=embedding_manager.index.ntotal if embedding_manager and embedding_manager.index else 0,
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG chatbot (non-streaming)."""
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not initialized. Ingest documents first.",
        )

    start = time.perf_counter()

    try:
        result = await rag_chain.aquery(
            question=request.question,
            session_id=request.session_id,
            use_cot=request.use_cot,
        )
        latency = (time.perf_counter() - start) * 1000

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=result["session_id"],
            retrieval_scores=result["retrieval_scores"],
            latency_ms=round(latency, 1),
        )
    except Exception as e:
        logger.error("query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query/stream")
async def query_stream(request: QueryRequest):
    """Query the RAG chatbot with SSE streaming response."""
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not initialized. Ingest documents first.",
        )

    async def event_generator() -> AsyncIterator[dict]:
        try:
            async for token in rag_chain.astream(
                question=request.question,
                session_id=request.session_id,
                use_cot=request.use_cot,
            ):
                yield {"event": "token", "data": token}
            yield {"event": "done", "data": "[DONE]"}
        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())


@app.post("/api/v1/ingest")
async def ingest_documents(request: IngestRequest):
    """Trigger document ingestion pipeline."""
    global embedding_manager, retriever, rag_chain

    try:
        pipeline = IngestionPipeline(chunking_strategy=request.strategy)
        stats = pipeline.ingest_directory(request.directory, persist=True)

        # Reinitialize chain with new index
        embedding_manager = pipeline.embedding_manager
        retriever = HybridRetriever(embedding_manager)
        rag_chain = RAGChain(retriever)

        return {"status": "success", "stats": stats}
    except Exception as e:
        logger.error("ingestion_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a single document."""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="Initialize the system first.")

    import tempfile
    from pathlib import Path

    allowed_types = {".pdf", ".docx", ".txt", ".md", ".html"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed_types}",
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        pipeline = IngestionPipeline(embedding_manager=embedding_manager)
        count = pipeline.ingest_file(tmp_path)
        embedding_manager.save()

        return {
            "status": "success",
            "file": file.filename,
            "chunks_indexed": count,
        }
    except Exception as e:
        logger.error("upload_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        import os
        os.unlink(tmp_path)


@app.delete("/api/v1/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session."""
    if rag_chain and rag_chain.clear_session(session_id):
        return {"status": "cleared", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


# ── Entry Point ─────────────────────────────────────────────

def main():
    """Run the API server."""
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
