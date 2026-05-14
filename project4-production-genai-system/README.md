# Project 4: Production GenAI System

> Full-stack production GenAI platform with RAG, Agents, Auth, Cloud Infrastructure (AWS/Azure), Kubernetes, Terraform, Helm, and comprehensive monitoring.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          CLOUD INFRASTRUCTURE                            │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │                     KUBERNETES (EKS / AKS)                      │     │
│  │                                                                 │     │
│  │  ┌─────────┐  ┌─────────────┐  ┌──────────┐  ┌────────────┐   │     │
│  │  │  NGINX  │  │  API Gateway │  │  Auth    │  │  Rate      │   │     │
│  │  │  Ingress│──│  (FastAPI)   │──│  (JWT+   │  │  Limiter   │   │     │
│  │  └─────────┘  └──────┬──────┘  │  OAuth2) │  └────────────┘   │     │
│  │                       │         └──────────┘                    │     │
│  │         ┌─────────────┼─────────────┐                          │     │
│  │         ▼             ▼             ▼                          │     │
│  │  ┌────────────┐ ┌──────────┐ ┌──────────────┐                 │     │
│  │  │ RAG Service│ │  Agent   │ │  LLM Serving │                 │     │
│  │  │ (LangChain)│ │  Service │ │  (vLLM/TGI)  │                 │     │
│  │  └─────┬──────┘ │(LangGraph│ └──────────────┘                 │     │
│  │        │        └──────────┘                                   │     │
│  │  ┌─────▼────────────────────────────┐                          │     │
│  │  │  Vector DB (Pinecone / pgvector) │                          │     │
│  │  └─────────────────────────────────┘                           │     │
│  │                                                                 │     │
│  │  ┌──────────────────────────────────────────────────────┐      │     │
│  │  │        OBSERVABILITY STACK                            │      │     │
│  │  │  Prometheus │ Grafana │ Loki │ OpenTelemetry          │      │     │
│  │  └──────────────────────────────────────────────────────┘      │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │  S3 / Blob   │  │  RDS / SQL   │  │  Secrets     │                   │
│  │  (Documents) │  │  (Metadata)  │  │  Manager     │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │          TERRAFORM (IaC) — VPC, EKS, RDS, S3, IAM        │           │
│  └──────────────────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Cloud | AWS (EKS, S3, RDS, Secrets Manager) |
| IaC | Terraform (modular, multi-env) |
| Orchestration | Kubernetes + Helm |
| API Gateway | FastAPI + NGINX Ingress |
| Auth | JWT + OAuth2 + API Key |
| RAG | LangChain + Pinecone/pgvector |
| Agents | LangGraph multi-agent |
| LLM | OpenAI GPT-4o / Self-hosted vLLM |
| Monitoring | Prometheus + Grafana + Loki + OTEL |
| CI/CD | GitHub Actions (multi-env) |
| Security | RBAC, Network Policies, Pod Security |
