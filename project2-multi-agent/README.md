# Project 2: Multi-Agent Research Assistant

> Production multi-agent system using LangGraph, CrewAI, and vector databases for autonomous research workflows.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR (LangGraph)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Research  │  │ Analysis │  │  Writer  │  │   Reviewer    │   │
│  │  Agent    │  │  Agent   │  │  Agent   │  │    Agent      │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘   │
│       │              │              │               │           │
│  ┌────▼──────────────▼──────────────▼───────────────▼──────┐   │
│  │                    TOOL REGISTRY                         │   │
│  │  Web Search │ ArXiv │ Wikipedia │ Calculator │ Code Exec │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │              SHARED MEMORY (ChromaDB / FAISS)            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Agents

| Agent | Role | Tools |
|-------|------|-------|
| **Research Agent** | Gather information from multiple sources | Web Search, ArXiv, Wikipedia |
| **Analysis Agent** | Synthesize findings, identify patterns | Calculator, Code Executor |
| **Writer Agent** | Produce structured research reports | Markdown Formatter |
| **Reviewer Agent** | Quality check, fact verification | Web Search, Memory Lookup |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph + CrewAI |
| LLM | OpenAI GPT-4o / Anthropic Claude |
| Vector Store | ChromaDB (dev), Pinecone (prod) |
| Memory | Shared agent memory with ChromaDB |
| API | FastAPI + WebSocket |
| Testing | pytest + agent evaluation framework |
| CI/CD | GitHub Actions multi-environment |
