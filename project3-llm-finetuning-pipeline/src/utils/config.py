"""
Configuration for LLM Fine-Tuning Pipeline.
Supports model configs, training hyperparams, and multi-environment deployment.
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
    app_name: str = "llm-finetuning-pipeline"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # --- Model ---
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_revision: str = "main"
    tokenizer_name: Optional[str] = None
    max_seq_length: int = 2048
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # --- LoRA ---
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    use_qlora: bool = True
    qlora_bits: int = 4
    qlora_compute_dtype: str = "bfloat16"
    qlora_quant_type: str = "nf4"
    use_double_quant: bool = True

    # --- Training ---
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    seed: int = 42

    # --- Dataset ---
    dataset_name: str = ""
    dataset_split: str = "train"
    dataset_text_field: str = "text"
    dataset_max_samples: int = -1
    val_split_ratio: float = 0.05

    # --- DeepSpeed ---
    use_deepspeed: bool = False
    deepspeed_config: str = "configs/models/deepspeed_zero3.json"

    # --- DDP ---
    use_ddp: bool = False
    ddp_backend: str = "nccl"
    local_rank: int = -1

    # --- MLflow ---
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "llm-finetuning"
    mlflow_run_name: str = ""

    # --- W&B ---
    wandb_project: str = "llm-finetuning"
    wandb_entity: str = ""
    wandb_api_key: str = ""
    use_wandb: bool = False

    # --- Serving ---
    serving_engine: str = "vllm"
    serving_host: str = "0.0.0.0"
    serving_port: int = 8002
    serving_max_model_len: int = 4096
    serving_gpu_memory_utilization: float = 0.85
    serving_tensor_parallel_size: int = 1

    # --- Evaluation ---
    eval_benchmarks: str = "mmlu,hellaswag,arc_challenge"
    eval_num_fewshot: int = 5

    # --- HuggingFace ---
    hf_token: str = ""
    push_to_hub: bool = False
    hub_model_id: str = ""

    @property
    def lora_target_modules_list(self) -> list[str]:
        return [m.strip() for m in self.lora_target_modules.split(",")]

    @property
    def eval_benchmarks_list(self) -> list[str]:
        return [b.strip() for b in self.eval_benchmarks.split(",")]


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
