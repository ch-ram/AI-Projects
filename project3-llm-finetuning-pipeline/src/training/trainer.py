"""
LLM Fine-Tuning Trainer with LoRA/QLoRA, DeepSpeed, and DDP support.
Integrates MLflow and W&B for experiment tracking.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import DatasetDict
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Unified experiment tracking with MLflow and W&B."""

    def __init__(self):
        settings = get_settings()
        self._mlflow = None
        self._wandb = None

        # MLflow setup
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment)
            self._mlflow = mlflow
            logger.info("mlflow_initialized", uri=settings.mlflow_tracking_uri)
        except Exception as e:
            logger.warning("mlflow_init_failed", error=str(e))

        # W&B setup
        if settings.use_wandb and settings.wandb_api_key:
            try:
                import wandb
                wandb.login(key=settings.wandb_api_key)
                self._wandb = wandb
                logger.info("wandb_initialized", project=settings.wandb_project)
            except Exception as e:
                logger.warning("wandb_init_failed", error=str(e))

    def start_run(self, run_name: str, config: dict) -> None:
        if self._mlflow:
            self._mlflow.start_run(run_name=run_name)
            self._mlflow.log_params(
                {k: str(v)[:250] for k, v in config.items()}
            )
        if self._wandb:
            settings = get_settings()
            self._wandb.init(
                project=settings.wandb_project,
                name=run_name,
                config=config,
            )

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        if self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)
        if self._wandb:
            self._wandb.log(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        if self._mlflow:
            self._mlflow.log_artifact(path)

    def end_run(self) -> None:
        if self._mlflow:
            self._mlflow.end_run()
        if self._wandb:
            self._wandb.finish()


class LLMTrainer:
    """
    Production LLM fine-tuning trainer.

    Features:
    - LoRA and QLoRA (4-bit quantized) fine-tuning
    - DeepSpeed ZeRO-2/3 distributed training
    - PyTorch DDP multi-GPU training
    - MLflow + W&B experiment tracking
    - Gradient checkpointing for memory efficiency
    - Mixed precision (bf16/fp16) training
    """

    def __init__(self):
        self.settings = get_settings()
        self.tracker = ExperimentTracker()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    def load_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load base model with optional quantization (QLoRA)."""
        s = self.settings

        logger.info(
            "loading_model",
            model=s.base_model,
            qlora=s.use_qlora,
            bits=s.qlora_bits,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            s.tokenizer_name or s.base_model,
            token=s.hf_token or None,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Quantization config for QLoRA
        bnb_config = None
        if s.use_qlora:
            compute_dtype = getattr(torch, s.qlora_compute_dtype, torch.bfloat16)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=(s.qlora_bits == 4),
                load_in_8bit=(s.qlora_bits == 8),
                bnb_4bit_quant_type=s.qlora_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=s.use_double_quant,
            )

        # Load model
        torch_dtype = getattr(torch, s.torch_dtype, torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            s.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto" if not s.use_ddp else None,
            attn_implementation=s.attn_implementation,
            token=s.hf_token or None,
            trust_remote_code=True,
        )

        # Prepare for k-bit training
        if s.use_qlora:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=s.gradient_checkpointing,
            )

        logger.info(
            "model_loaded",
            params=f"{self.model.num_parameters() / 1e9:.2f}B",
            dtype=str(torch_dtype),
        )

        return self.model, self.tokenizer

    def apply_lora(self) -> PeftModel:
        """Apply LoRA adapters to the base model."""
        s = self.settings

        lora_config = LoraConfig(
            r=s.lora_r,
            lora_alpha=s.lora_alpha,
            lora_dropout=s.lora_dropout,
            target_modules=s.lora_target_modules_list,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            inference_mode=False,
        )

        self.model = get_peft_model(self.model, lora_config)

        trainable, total = self.model.get_nb_trainable_parameters()
        pct = 100 * trainable / total

        logger.info(
            "lora_applied",
            r=s.lora_r,
            alpha=s.lora_alpha,
            trainable_params=f"{trainable:,}",
            total_params=f"{total:,}",
            trainable_pct=f"{pct:.2f}%",
        )

        return self.model

    def get_training_args(self) -> TrainingArguments:
        """Build TrainingArguments from settings."""
        s = self.settings

        args = TrainingArguments(
            output_dir=s.output_dir,
            num_train_epochs=s.num_train_epochs,
            per_device_train_batch_size=s.per_device_train_batch_size,
            per_device_eval_batch_size=s.per_device_eval_batch_size,
            gradient_accumulation_steps=s.gradient_accumulation_steps,
            learning_rate=s.learning_rate,
            weight_decay=s.weight_decay,
            warmup_ratio=s.warmup_ratio,
            lr_scheduler_type=s.lr_scheduler_type,
            max_grad_norm=s.max_grad_norm,
            fp16=s.fp16,
            bf16=s.bf16,
            gradient_checkpointing=s.gradient_checkpointing,
            optim=s.optim,
            logging_steps=s.logging_steps,
            save_steps=s.save_steps,
            eval_strategy="steps",
            eval_steps=s.eval_steps,
            save_total_limit=s.save_total_limit,
            seed=s.seed,
            report_to=self._get_report_to(),
            run_name=s.mlflow_run_name or f"{s.base_model.split('/')[-1]}-lora-r{s.lora_r}",
            # DeepSpeed
            deepspeed=s.deepspeed_config if s.use_deepspeed else None,
            # DDP
            ddp_backend=s.ddp_backend if s.use_ddp else None,
            local_rank=s.local_rank if s.use_ddp else -1,
            # Push to Hub
            push_to_hub=s.push_to_hub,
            hub_model_id=s.hub_model_id if s.push_to_hub else None,
            hub_token=s.hf_token if s.push_to_hub else None,
            # Misc
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
        )

        return args

    def train(self, dataset: DatasetDict) -> dict:
        """
        Execute the full fine-tuning workflow.

        Steps:
        1. Load base model
        2. Apply LoRA adapters
        3. Build training arguments
        4. Train with HuggingFace Trainer
        5. Save model and adapters
        6. Log metrics and artifacts
        """
        s = self.settings
        run_name = s.mlflow_run_name or f"{s.base_model.split('/')[-1]}-lora"

        # Track experiment
        config = {
            "base_model": s.base_model,
            "lora_r": s.lora_r,
            "lora_alpha": s.lora_alpha,
            "learning_rate": s.learning_rate,
            "epochs": s.num_train_epochs,
            "batch_size": s.per_device_train_batch_size,
            "grad_accum": s.gradient_accumulation_steps,
            "max_seq_length": s.max_seq_length,
            "use_qlora": s.use_qlora,
        }
        self.tracker.start_run(run_name, config)

        try:
            # Step 1: Load model
            self.load_model()

            # Step 2: Apply LoRA
            self.apply_lora()

            # Step 3: Training arguments
            training_args = self.get_training_args()

            # Step 4: Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

            # Step 5: Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset.get("validation"),
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )

            logger.info("training_start", config=config)

            # Step 6: Train
            result = trainer.train()

            # Step 7: Save
            save_path = Path(s.output_dir) / "final"
            trainer.save_model(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))

            # Step 8: Log metrics
            metrics = {
                "train_loss": result.training_loss,
                "train_runtime": result.metrics.get("train_runtime", 0),
                "train_samples_per_second": result.metrics.get(
                    "train_samples_per_second", 0
                ),
            }
            self.tracker.log_metrics(metrics)
            self.tracker.log_artifact(str(save_path))

            logger.info("training_complete", metrics=metrics)
            return metrics

        finally:
            self.tracker.end_run()

    def _get_report_to(self) -> list[str]:
        reporters = []
        if self.settings.mlflow_tracking_uri:
            reporters.append("mlflow")
        if self.settings.use_wandb:
            reporters.append("wandb")
        return reporters or ["none"]

    def merge_and_save(self, output_path: str) -> None:
        """Merge LoRA adapters into base model and save full model."""
        if self.model is None:
            raise ValueError("No model loaded")

        logger.info("merging_lora_adapters")
        merged = self.model.merge_and_unload()
        merged.save_pretrained(output_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_path)
        logger.info("merged_model_saved", path=output_path)
