# Project 3: LLM Fine-Tuning & MLOps Pipeline

> End-to-end pipeline for fine-tuning LLMs with LoRA/QLoRA, distributed training, experiment tracking, and model serving.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LLM FINE-TUNING PIPELINE                       │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌────────────┐  │
│  │   Data    │──▶│   Training   │──▶│   Eval   │──▶│  Serving   │  │
│  │ Pipeline  │   │  (LoRA/QLoRA)│   │  Suite   │   │  (vLLM)    │  │
│  └──────────┘   └──────────────┘   └──────────┘   └────────────┘  │
│       │               │                 │               │          │
│  ┌────▼────────────────▼─────────────────▼───────────────▼──────┐  │
│  │                    MLflow Experiment Tracking                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │            Docker + K8s + CI/CD (GitHub Actions)             │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Base Models | LLaMA 3.1, Mistral, Phi-3 |
| Fine-tuning | LoRA, QLoRA (PEFT + bitsandbytes) |
| Training | PyTorch + HuggingFace Transformers |
| Distributed | DeepSpeed ZeRO-3, PyTorch DDP |
| Experiment Tracking | MLflow + Weights & Biases |
| Evaluation | lm-eval-harness, custom benchmarks |
| Serving | vLLM, TGI, ONNX Runtime |
| Orchestration | Ray Train |
| CI/CD | GitHub Actions, Docker |
| Monitoring | Prometheus + Grafana |

## Features

- **Multi-GPU training** with DDP and DeepSpeed ZeRO-3
- **QLoRA** with 4-bit quantization for memory-efficient fine-tuning
- **MLflow** experiment tracking with artifact logging
- **Automated evaluation** on standard benchmarks
- **Model serving** with vLLM for high-throughput inference
- **A/B testing** framework for model comparison
- **CI/CD** with automated training, eval, and deployment gates
