"""
Agent definitions for Multi-Agent Research Assistant.
Each agent has a specialized role, persona, tools, and instructions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    name: str
    role: str
    goal: str
    backstory: str
    tools: list[str] = field(default_factory=list)
    model: Optional[str] = None
    temperature: float = 0.1
    max_iterations: int = 10


# ── Agent Definitions ──────────────────────────────────────

RESEARCH_AGENT = AgentConfig(
    name="research_agent",
    role="Senior Research Analyst",
    goal="Gather comprehensive, accurate information from multiple authoritative sources",
    backstory="""You are a senior research analyst with 15 years of experience in 
    academic and industry research. You excel at finding relevant information from 
    multiple sources, cross-referencing facts, and identifying the most authoritative 
    and recent sources. You always cite your sources and note when information might 
    be outdated or conflicting.""",
    tools=["web_search", "arxiv_search", "wikipedia_search"],
    temperature=0.1,
)

ANALYSIS_AGENT = AgentConfig(
    name="analysis_agent",
    role="Data Analysis Specialist",
    goal="Synthesize research findings into structured insights with quantitative support",
    backstory="""You are a data-driven analyst who specializes in synthesizing 
    large volumes of information into actionable insights. You identify patterns, 
    trends, and anomalies in data. You use quantitative methods when applicable 
    and always distinguish between correlation and causation. You present findings 
    with appropriate confidence levels.""",
    tools=["calculator", "code_executor"],
    temperature=0.0,
)

WRITER_AGENT = AgentConfig(
    name="writer_agent",
    role="Technical Writer & Report Author",
    goal="Produce well-structured, clear, and comprehensive research reports",
    backstory="""You are an expert technical writer with experience writing for 
    both academic journals and industry publications. You structure information 
    logically, use clear language, and ensure every claim is supported by evidence. 
    You format reports with proper headings, citations, and executive summaries.""",
    tools=["markdown_formatter"],
    temperature=0.3,
)

REVIEWER_AGENT = AgentConfig(
    name="reviewer_agent",
    role="Quality Assurance Reviewer",
    goal="Ensure accuracy, completeness, and quality of research outputs",
    backstory="""You are a meticulous reviewer with expertise in fact-checking 
    and quality assurance. You verify claims against sources, check for logical 
    consistency, identify gaps in analysis, and ensure the final output meets 
    professional standards. You provide specific, actionable feedback.""",
    tools=["web_search", "memory_lookup"],
    temperature=0.0,
)

ALL_AGENTS = [RESEARCH_AGENT, ANALYSIS_AGENT, WRITER_AGENT, REVIEWER_AGENT]


class Agent:
    """
    Wrapper around an LLM with agent-specific persona and capabilities.
    """

    def __init__(self, config: AgentConfig, tools: Optional[list] = None):
        settings = get_settings()
        self.config = config
        self.llm = ChatOpenAI(
            model=config.model or settings.primary_model,
            temperature=config.temperature,
            max_tokens=settings.max_tokens,
        )
        self.tools = tools or []

        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

        self._system_prompt = self._build_system_prompt()
        logger.info("agent_initialized", name=config.name, role=config.role)

    def _build_system_prompt(self) -> str:
        return f"""## Role
{self.config.role}

## Goal
{self.config.goal}

## Background
{self.config.backstory}

## Instructions
1. Focus on your specific role and goal.
2. Use your available tools when they can help accomplish the task.
3. Be thorough but concise in your responses.
4. Always cite sources and note confidence levels.
5. If you cannot complete a task, explain why and suggest alternatives.
6. Collaborate with other agents by providing clear, structured outputs."""

    async def ainvoke(self, task: str, context: str = "") -> str:
        """Execute a task with optional context from other agents."""
        messages = [
            SystemMessage(content=self._system_prompt),
        ]
        if context:
            messages.append(HumanMessage(
                content=f"## Context from Previous Agents\n{context}\n\n## Your Task\n{task}"
            ))
        else:
            messages.append(HumanMessage(content=task))

        logger.info("agent_executing", agent=self.config.name, task=task[:100])

        response = await self.llm_with_tools.ainvoke(messages)

        logger.info(
            "agent_completed",
            agent=self.config.name,
            output_length=len(response.content),
        )
        return response.content

    def invoke(self, task: str, context: str = "") -> str:
        """Synchronous task execution."""
        messages = [
            SystemMessage(content=self._system_prompt),
        ]
        if context:
            messages.append(HumanMessage(
                content=f"## Context from Previous Agents\n{context}\n\n## Your Task\n{task}"
            ))
        else:
            messages.append(HumanMessage(content=task))

        response = self.llm_with_tools.invoke(messages)
        return response.content
