"""
Configuration for Multi-Agent Research Assistant.
"""
from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    QA = "qa"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Environment = Environment.DEVELOPMENT
    app_name: str = "multi-agent-research"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    primary_model: str = "gpt-4o"
    fast_model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4096

    # Vector DB
    vector_db_provider: str = "chromadb"  # chromadb | pinecone | faiss
    chromadb_path: str = "./data/chromadb"
    pinecone_api_key: str = ""
    pinecone_index: str = "research-memory"
    pinecone_environment: str = "us-east-1"

    # Tools
    tavily_api_key: str = ""
    max_search_results: int = 5
    max_agent_iterations: int = 15

    # Orchestrator
    max_concurrent_agents: int = 4
    agent_timeout_seconds: int = 120
    enable_human_in_loop: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    api_workers: int = 1
    cors_origins: str = "http://localhost:3000"

    # Observability
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "multi-agent-research"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


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
