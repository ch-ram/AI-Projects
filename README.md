# 🚀 Production AI/ML Portfolio — Parsharam Chinthalathadem (Ram)

> **4 Industry-Grade Projects** covering Generative AI, LLMs, NLP, Agentic AI, Deep Learning, MLOps, and Cloud System Design.

---

## Portfolio Architecture

```
ai-portfolio/
├── project1-rag-chatbot/          # Document Q&A RAG Chatbot
├── project2-multi-agent/          # Multi-Agent Research Assistant
├── project3-llm-finetuning-pipeline/  # LLM Fine-Tuning + MLOps Pipeline
├── project4-production-genai-system/  # Production GenAI System (Full Stack)
```

## Projects Overview

| # | Project | Stack | Key Skills |
|---|---------|-------|------------|
| 1 | **Document Q&A RAG Chatbot** | LangChain, OpenAI, FAISS, FastAPI | RAG, Embeddings, Prompt Engineering, Tokenization |
| 2 | **Multi-Agent Research Assistant** | LangGraph, CrewAI, ChromaDB, Pinecone | Agentic AI, Multi-Agent Orchestration, Tool Use |
| 3 | **LLM Fine-Tuning Pipeline** | PyTorch, DeepSpeed, LoRA/QLoRA, MLflow | DDP, Experiment Tracking, Model Serving, CI/CD |
| 4 | **Production GenAI System** | AWS/Azure, Kubernetes, Terraform, Helm | Cloud Deployment, Monitoring, Security, E2E Design |

## Git Branching Strategy

All projects follow **GitFlow** with environment-mapped branches:

```
main (prod)  ←  release/*  ←  develop  ←  feature/*
                                ↑
                            hotfix/*
```

| Branch | Environment | Purpose |
|--------|-------------|---------|
| `main` | **Production** | Stable, tagged releases only |
| `release/*` | **Staging** | Pre-prod validation |
| `develop` | **QA/Test** | Integration branch |
| `feature/*` | **Dev** | Individual feature work |
| `hotfix/*` | **Production** | Emergency fixes |

## CI/CD Pipeline

Each project includes GitHub Actions workflows:
- **ci.yml** — Lint, test, security scan on every PR
- **cd-dev.yml** — Auto-deploy to dev on feature branch push
- **cd-staging.yml** — Deploy to staging on release/* branch
- **cd-prod.yml** — Deploy to prod on main (manual approval gate)

## Quick Start

```bash
# Clone
git clone https://github.com/ch-ram/ai-portfolio.git
cd ai-portfolio

# Pick a project
cd project1-rag-chatbot

# Setup
cp configs/dev/.env.dev .env
pip install -r requirements.txt

# Run
python -m src.api.main
```

## Author

**Parsharam Chinthalathadem (Ram)**
M.S. Computer Science (AI/ML) — Auburn University at Montgomery
CEO & Founder — PSR IT Solutions LLC
