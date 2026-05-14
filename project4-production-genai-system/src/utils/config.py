"""
Configuration for Production GenAI System.
Multi-service, multi-environment settings.
"""
from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    QA = "qa"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    app_env: Environment = Environment.DEVELOPMENT
    app_name: str = "genai-platform"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # --- LLM ---
    openai_api_key: str = ""
    primary_model: str = "gpt-4o"
    fast_model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4096

    # --- RAG ---
    vector_db_provider: str = "pinecone"
    pinecone_api_key: str = ""
    pinecone_index: str = "genai-prod"
    pinecone_environment: str = "us-east-1"
    pgvector_url: str = ""
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5

    # --- Auth ---
    jwt_secret_key: str = "CHANGE-ME-IN-PRODUCTION"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    api_key_header: str = "X-API-Key"
    oauth2_enabled: bool = False
    oauth2_provider_url: str = ""
    oauth2_client_id: str = ""
    oauth2_client_secret: str = ""

    # --- Database ---
    database_url: str = "postgresql://user:pass@localhost:5432/genai"
    redis_url: str = "redis://localhost:6379/0"

    # --- AWS ---
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket: str = "genai-documents"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    cors_origins: str = "http://localhost:3000"
    rate_limit_rpm: int = 60

    # --- Observability ---
    otel_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "genai-platform"
    enable_prometheus: bool = True
    enable_tracing: bool = True
    grafana_url: str = "http://localhost:3000"
    loki_url: str = "http://localhost:3100"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION


@lru_cache()
def get_settings() -> Settings:
    env = os.getenv("APP_ENV", "development").lower()
    env_map = {
        "development": "configs/dev/.env.dev",
        "qa": "configs/qa/.env.qa",
        "staging": "configs/staging/.env.staging",
        "production": "configs/prod/.env.prod",
    }
    env_file = env_map.get(env, ".env")
    if Path(env_file).exists():
        return Settings(_env_file=env_file)
    return Settings()
