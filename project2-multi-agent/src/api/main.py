"""
FastAPI application for Multi-Agent Research Assistant.
Supports REST and WebSocket for real-time agent status updates.
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.orchestrator.graph import ResearchOrchestrator
from src.utils.config import get_settings
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

orchestrator: Optional[ResearchOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    setup_logging()
    orchestrator = ResearchOrchestrator()
    logger.info("multi_agent_api_ready")
    yield
    logger.info("multi_agent_api_shutdown")


settings = get_settings()

app = FastAPI(
    title="Multi-Agent Research Assistant API",
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


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=5000)
    max_revisions: int = Field(2, ge=0, le=5)


class ResearchResponse(BaseModel):
    task_id: str
    query: str
    report: str
    status: str
    revisions: int
    latency_seconds: float


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": settings.app_version,
        "agents": ["research", "analysis", "writer", "reviewer"],
    }


@app.post("/api/v1/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    """Execute a full multi-agent research workflow."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")

    task_id = str(uuid4())[:8]
    start = time.perf_counter()

    try:
        result = await orchestrator.research(
            query=request.query,
            max_revisions=request.max_revisions,
            thread_id=task_id,
        )
        elapsed = time.perf_counter() - start

        return ResearchResponse(
            task_id=task_id,
            query=request.query,
            report=result["report"],
            status=result["status"],
            revisions=result["revisions"],
            latency_seconds=round(elapsed, 2),
        )
    except Exception as e:
        logger.error("research_failed", error=str(e), task_id=task_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/research")
async def websocket_research(websocket: WebSocket):
    """WebSocket endpoint for real-time research with agent status updates."""
    await websocket.accept()
    logger.info("websocket_connected")

    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            query = request.get("query", "")

            if not query:
                await websocket.send_json({"error": "Query required"})
                continue

            await websocket.send_json({
                "event": "started",
                "agent": "orchestrator",
                "message": f"Starting research: {query[:100]}",
            })

            try:
                result = await orchestrator.research(query=query)

                await websocket.send_json({
                    "event": "complete",
                    "report": result["report"],
                    "status": result["status"],
                    "revisions": result["revisions"],
                })
            except Exception as e:
                await websocket.send_json({
                    "event": "error",
                    "message": str(e),
                })

    except WebSocketDisconnect:
        logger.info("websocket_disconnected")


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
