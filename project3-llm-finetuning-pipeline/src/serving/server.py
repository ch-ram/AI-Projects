"""
Production model serving with vLLM for high-throughput inference.
Supports OpenAI-compatible API, batching, and model registry.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils.config import get_settings
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# ── Request/Response Models ─────────────────────────────────

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str | list[str] = ""
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[list[str]] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stream: bool = False


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


# ── Model Registry ──────────────────────────────────────────

class ModelRegistry:
    """Track and manage fine-tuned model versions."""

    def __init__(self, registry_path: str = "./models/registry.json"):
        self.path = Path(registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                self._registry = json.load(f)

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def register(
        self,
        model_id: str,
        model_path: str,
        base_model: str,
        metrics: dict,
        metadata: Optional[dict] = None,
    ) -> None:
        self._registry[model_id] = {
            "model_path": model_path,
            "base_model": base_model,
            "metrics": metrics,
            "metadata": metadata or {},
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "ready",
        }
        self._save()
        logger.info("model_registered", model_id=model_id)

    def get(self, model_id: str) -> Optional[dict]:
        return self._registry.get(model_id)

    def list_models(self) -> list[dict]:
        return [
            {"model_id": k, **v}
            for k, v in self._registry.items()
        ]

    def get_best_model(self, metric: str = "eval_loss") -> Optional[str]:
        """Get model ID with the best metric score."""
        best_id, best_score = None, float("inf")
        for mid, info in self._registry.items():
            score = info.get("metrics", {}).get(metric, float("inf"))
            if score < best_score:
                best_score = score
                best_id = mid
        return best_id


# ── vLLM Serving Engine ────────────────────────────────────

class ServingEngine:
    """
    Production LLM serving with vLLM.

    Features:
    - OpenAI-compatible API
    - Continuous batching
    - PagedAttention for efficient memory
    - Tensor parallelism for multi-GPU
    """

    def __init__(self, model_path: Optional[str] = None):
        settings = get_settings()
        self.model_path = model_path or settings.output_dir + "/final"
        self.engine = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize vLLM engine."""
        settings = get_settings()

        try:
            from vllm import LLM, SamplingParams

            self.engine = LLM(
                model=self.model_path,
                tokenizer=self.model_path,
                max_model_len=settings.serving_max_model_len,
                gpu_memory_utilization=settings.serving_gpu_memory_utilization,
                tensor_parallel_size=settings.serving_tensor_parallel_size,
                dtype="auto",
                trust_remote_code=True,
            )
            self._initialized = True
            logger.info("vllm_engine_initialized", model=self.model_path)

        except ImportError:
            logger.warning("vllm_not_installed_using_transformers_fallback")
            self._init_transformers_fallback()

    def _init_transformers_fallback(self) -> None:
        """Fallback to HuggingFace pipeline when vLLM is unavailable."""
        import torch
        from transformers import pipeline as hf_pipeline

        self._pipeline = hf_pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._initialized = True
        logger.info("transformers_fallback_initialized")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate text from a prompt."""
        if not self._initialized:
            self.initialize()

        if self.engine is not None:
            from vllm import SamplingParams
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            outputs = self.engine.generate([prompt], params)
            return outputs[0].outputs[0].text
        else:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )
            return outputs[0]["generated_text"][len(prompt):]

    def batch_generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> list[str]:
        """Batch generate for multiple prompts."""
        if not self._initialized:
            self.initialize()

        if self.engine is not None:
            from vllm import SamplingParams
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
            )
            outputs = self.engine.generate(prompts, params)
            return [o.outputs[0].text for o in outputs]
        else:
            results = []
            for prompt in prompts:
                results.append(self.generate(prompt, max_tokens, temperature))
            return results


# ── FastAPI Serving App ─────────────────────────────────────

def create_serving_app(model_path: Optional[str] = None) -> FastAPI:
    """Create FastAPI app for model serving."""
    app = FastAPI(title="LLM Serving API", version="0.1.0")
    engine = ServingEngine(model_path)

    @app.on_event("startup")
    async def startup():
        engine.initialize()

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model": engine.model_path,
            "initialized": engine._initialized,
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "data": [{"id": engine.model_path, "object": "model"}]
        }

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        try:
            text = engine.generate(
                prompt=request.prompt if isinstance(request.prompt, str) else request.prompt[0],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )
            return CompletionResponse(
                id=f"cmpl-{int(time.time())}",
                created=int(time.time()),
                model=engine.model_path,
                choices=[{"text": text, "index": 0, "finish_reason": "stop"}],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        prompt = "\n".join(
            f"<|{m.role}|>\n{m.content}" for m in request.messages
        ) + "\n<|assistant|>\n"

        try:
            text = engine.generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": engine.model_path,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
