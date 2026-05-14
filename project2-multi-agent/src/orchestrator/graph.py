"""
LangGraph-based multi-agent orchestrator.
Implements a directed graph of agent interactions with state management.

Workflow:
  START → Research Agent → Analysis Agent → Writer Agent → Reviewer Agent
    ↑                                                          │
    └──────────── (revision needed) ──────────────────────────┘
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.agents.definitions import (
    ANALYSIS_AGENT,
    RESEARCH_AGENT,
    REVIEWER_AGENT,
    WRITER_AGENT,
    Agent,
    AgentConfig,
)
from src.tools.registry import get_tools_for_agent
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── State Definition ───────────────────────────────────────

class ResearchState(TypedDict):
    """Shared state across all agents in the research workflow."""
    # Input
    query: str
    # Agent outputs (appended)
    research_output: Annotated[list[str], operator.add]
    analysis_output: Annotated[list[str], operator.add]
    draft_output: Annotated[list[str], operator.add]
    review_output: Annotated[list[str], operator.add]
    # Control flow
    revision_count: int
    max_revisions: int
    status: str
    final_report: str
    error: Optional[str]


# ── Agent Node Functions ───────────────────────────────────

def _create_agent(config: AgentConfig) -> Agent:
    """Factory to create an agent with resolved tools."""
    tools = get_tools_for_agent(config.tools)
    return Agent(config, tools)


async def research_node(state: ResearchState) -> dict:
    """Research agent: gather information from multiple sources."""
    logger.info("research_node_start", query=state["query"])
    agent = _create_agent(RESEARCH_AGENT)

    task = f"""Research the following topic thoroughly:

TOPIC: {state['query']}

Instructions:
1. Search the web for the latest information
2. Search ArXiv for relevant academic papers
3. Search Wikipedia for background context
4. Compile all findings with source citations
5. Identify any conflicting information between sources
6. Note the date and reliability of each source

Provide a comprehensive research summary with all sources cited."""

    try:
        output = await agent.ainvoke(task)
        return {
            "research_output": [output],
            "status": "research_complete",
        }
    except Exception as e:
        logger.error("research_node_failed", error=str(e))
        return {"error": str(e), "status": "failed"}


async def analysis_node(state: ResearchState) -> dict:
    """Analysis agent: synthesize research findings."""
    logger.info("analysis_node_start")
    agent = _create_agent(ANALYSIS_AGENT)

    research_context = "\n\n".join(state.get("research_output", []))

    task = f"""Analyze the following research findings and produce a structured analysis:

RESEARCH FINDINGS:
{research_context}

Instructions:
1. Identify key themes and patterns across sources
2. Highlight areas of consensus and disagreement
3. Quantify findings where possible
4. Assess the strength of evidence for each claim
5. Identify gaps in the research
6. Provide a confidence level (high/medium/low) for each key finding

Structure your analysis with clear sections and bullet points."""

    try:
        output = await agent.ainvoke(task, context=research_context)
        return {
            "analysis_output": [output],
            "status": "analysis_complete",
        }
    except Exception as e:
        logger.error("analysis_node_failed", error=str(e))
        return {"error": str(e), "status": "failed"}


async def writer_node(state: ResearchState) -> dict:
    """Writer agent: produce the research report."""
    logger.info("writer_node_start")
    agent = _create_agent(WRITER_AGENT)

    research = "\n\n".join(state.get("research_output", []))
    analysis = "\n\n".join(state.get("analysis_output", []))

    # Check if this is a revision
    revision_context = ""
    if state.get("review_output"):
        revision_context = f"""

## Reviewer Feedback (Address These Issues):
{state['review_output'][-1]}
"""

    task = f"""Write a comprehensive research report based on the analysis below.

ORIGINAL QUERY: {state['query']}

RESEARCH DATA:
{research}

ANALYSIS:
{analysis}
{revision_context}

Report Structure:
1. Executive Summary (2-3 paragraphs)
2. Introduction & Background
3. Key Findings (organized by theme)
4. Analysis & Discussion
5. Limitations & Gaps
6. Conclusions & Recommendations
7. References

Requirements:
- Every claim must cite its source
- Use clear, professional language
- Include quantitative data where available
- Distinguish between established facts and emerging findings"""

    try:
        output = await agent.ainvoke(task, context=f"{research}\n\n{analysis}")
        return {
            "draft_output": [output],
            "status": "draft_complete",
        }
    except Exception as e:
        logger.error("writer_node_failed", error=str(e))
        return {"error": str(e), "status": "failed"}


async def reviewer_node(state: ResearchState) -> dict:
    """Reviewer agent: quality check the report."""
    logger.info("reviewer_node_start", revision=state.get("revision_count", 0))
    agent = _create_agent(REVIEWER_AGENT)

    draft = state["draft_output"][-1] if state.get("draft_output") else ""

    task = f"""Review the following research report for quality, accuracy, and completeness.

ORIGINAL QUERY: {state['query']}

DRAFT REPORT:
{draft}

Review Criteria:
1. ACCURACY: Are all claims properly sourced? Any factual errors?
2. COMPLETENESS: Does it address all aspects of the query?
3. STRUCTURE: Is it well-organized and logical?
4. CLARITY: Is the writing clear and professional?
5. BALANCE: Are multiple perspectives represented fairly?

Provide your assessment in this format:
- VERDICT: APPROVE or NEEDS_REVISION
- SCORE: 1-10
- ISSUES: List any specific problems
- SUGGESTIONS: Specific improvements needed"""

    try:
        output = await agent.ainvoke(task)

        # Parse verdict
        needs_revision = "NEEDS_REVISION" in output.upper()
        status = "needs_revision" if needs_revision else "approved"

        return {
            "review_output": [output],
            "status": status,
            "revision_count": state.get("revision_count", 0) + 1,
        }
    except Exception as e:
        logger.error("reviewer_node_failed", error=str(e))
        return {"error": str(e), "status": "failed"}


def finalize_node(state: ResearchState) -> dict:
    """Finalize the research report."""
    final_report = state["draft_output"][-1] if state.get("draft_output") else ""
    return {
        "final_report": final_report,
        "status": "complete",
    }


# ── Router (Conditional Edge) ──────────────────────────────

def should_revise(state: ResearchState) -> Literal["revise", "finalize"]:
    """Decide whether the report needs revision or is ready."""
    if state.get("error"):
        return "finalize"
    if state.get("status") == "needs_revision":
        max_rev = state.get("max_revisions", 2)
        if state.get("revision_count", 0) < max_rev:
            logger.info("revision_needed", count=state["revision_count"])
            return "revise"
    return "finalize"


# ── Graph Builder ──────────────────────────────────────────

def build_research_graph() -> StateGraph:
    """
    Build the LangGraph research workflow.

    Graph:
        research → analysis → writer → reviewer → [revise|finalize]
                                  ↑                    │
                                  └────────────────────┘
    """
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("research", research_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("finalize", finalize_node)

    # Add edges
    graph.set_entry_point("research")
    graph.add_edge("research", "analysis")
    graph.add_edge("analysis", "writer")
    graph.add_edge("writer", "reviewer")

    # Conditional: revise or finalize
    graph.add_conditional_edges(
        "reviewer",
        should_revise,
        {
            "revise": "writer",
            "finalize": "finalize",
        },
    )
    graph.add_edge("finalize", END)

    return graph


class ResearchOrchestrator:
    """
    High-level orchestrator for multi-agent research workflows.
    """

    def __init__(self):
        self.graph = build_research_graph()
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)
        logger.info("orchestrator_initialized")

    async def research(
        self,
        query: str,
        max_revisions: int = 2,
        thread_id: Optional[str] = None,
    ) -> dict:
        """
        Execute a full research workflow.

        Args:
            query: Research topic
            max_revisions: Max revision cycles
            thread_id: Thread ID for checkpointing

        Returns:
            Final state with report and metadata
        """
        logger.info("research_start", query=query[:100])

        initial_state: ResearchState = {
            "query": query,
            "research_output": [],
            "analysis_output": [],
            "draft_output": [],
            "review_output": [],
            "revision_count": 0,
            "max_revisions": max_revisions,
            "status": "started",
            "final_report": "",
            "error": None,
        }

        config = {"configurable": {"thread_id": thread_id or "default"}}

        final_state = await self.app.ainvoke(initial_state, config)

        logger.info(
            "research_complete",
            status=final_state["status"],
            revisions=final_state["revision_count"],
        )

        return {
            "query": query,
            "report": final_state.get("final_report", ""),
            "status": final_state["status"],
            "revisions": final_state["revision_count"],
            "research_data": final_state.get("research_output", []),
            "analysis_data": final_state.get("analysis_output", []),
        }
