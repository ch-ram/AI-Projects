# Project 1: Document Q&A RAG Chatbot

> Production-grade Retrieval-Augmented Generation chatbot using LangChain, OpenAI, and FAISS.

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Frontend   │───▶│   FastAPI    │───▶│  RAG Engine  │───▶│   OpenAI     │
│  (Streamlit) │    │   Gateway    │    │  (LangChain) │    │   GPT-4o     │
└──────────────┘    └──────────────┘    └──────┬───────┘    └──────────────┘
                                               │
                    ┌──────────────┐    ┌──────▼───────┐
                    │  Document    │───▶│    FAISS      │
                    │  Ingestion   │    │  Vector Store │
                    │  Pipeline    │    └──────────────┘
                    └──────────────┘
```

## Key Features

- **Multi-format ingestion**: PDF, DOCX, TXT, Markdown, HTML
- **Chunking strategies**: Recursive, Semantic, Sentence-based
- **Hybrid retrieval**: Dense (FAISS) + Sparse (BM25) + Re-ranking
- **Prompt engineering**: System prompts, few-shot, chain-of-thought
- **Streaming responses**: Token-by-token SSE streaming
- **Conversation memory**: Sliding window + summary memory
- **Evaluation**: RAGAS metrics (faithfulness, relevance, context recall)
- **Observability**: LangSmith tracing, structured logging

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Embeddings | text-embedding-3-large (3072d) |
| Vector Store | FAISS (IVFFlat + HNSW) |
| Framework | LangChain 0.3+ |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Testing | pytest + RAGAS |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |

## Quick Start

```bash
cp configs/dev/.env.dev .env
pip install -r requirements.txt
python -m src.ingestion.pipeline --input-dir ./data/documents
python -m src.api.main
```

## Project Structure

```
project1-rag-chatbot/
├── src/
│   ├── ingestion/          # Document loading, chunking, embedding
│   ├── retrieval/          # FAISS, BM25, hybrid retriever, re-ranker
│   ├── generation/         # Prompt templates, LLM chains, memory
│   ├── api/                # FastAPI endpoints, middleware, SSE
│   └── utils/              # Logging, config, helpers
├── configs/
│   ├── dev/                # Dev environment config
│   ├── qa/                 # QA environment config
│   ├── staging/            # Staging environment config
│   └── prod/               # Production environment config
├── tests/
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-end tests
├── scripts/                # Setup, seed, benchmark scripts
├── docker/                 # Dockerfiles and compose
├── .github/workflows/      # CI/CD pipelines
├── monitoring/             # Health checks, metrics
└── docs/                   # Architecture, API docs
```
