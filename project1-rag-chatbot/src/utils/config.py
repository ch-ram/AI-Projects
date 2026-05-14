"""
Centralized configuration management with Pydantic Settings.
Supports multi-environment loading (dev/qa/staging/prod).
"""
from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    QA = "qa"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_env: Environment = Environment.DEVELOPMENT
    app_name: str = "rag-chatbot"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dimensions: int = 3072
    openai_max_tokens: int = 4096
    openai_temperature: float = 0.1

    # --- FAISS ---
    faiss_index_path: str = "./data/faiss_index"
    faiss_index_type: str = "IVFFlat"
    faiss_nlist: int = 100
    faiss_nprobe: int = 10

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: str = "recursive"

    # --- Retrieval ---
    retrieval_top_k: int = 5
    retrieval_strategy: str = "hybrid"
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    reranker_enabled: bool = True
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    reranker_top_k: int = 3

    # --- Memory ---
    memory_type: str = "sliding_window"
    memory_window_size: int = 10
    memory_summary_enabled: bool = True

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    cors_origins: str = "http://localhost:3000"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 3600

    # --- Observability ---
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-chatbot"
    enable_prometheus: bool = True
    prometheus_port: int = 9090

    # --- Rate Limiting ---
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: str) -> str:
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION


@lru_cache()
def get_settings() -> Settings:
    """Singleton settings loader with environment detection."""
    env = os.getenv("APP_ENV", "development").lower()
    env_file_map = {
        "development": "configs/dev/.env.dev",
        "qa": "configs/qa/.env.qa",
        "staging": "configs/staging/.env.staging",
        "production": "configs/prod/.env.prod",
    }
    env_file = env_file_map.get(env, ".env")
    if Path(env_file).exists():
        return Settings(_env_file=env_file)
    return Settings()
