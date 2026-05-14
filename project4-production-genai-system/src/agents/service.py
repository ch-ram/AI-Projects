"""
Production agent service with task routing and execution.
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AgentService:
    """
    Multi-purpose agent service supporting:
    - Task routing (classify → delegate)
    - Tool-augmented generation
    - Multi-step reasoning
    """

    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=settings.primary_model,
            temperature=settings.temperature,
        )
        self.fast_llm = ChatOpenAI(
            model=settings.fast_model,
            temperature=0.0,
        )
        logger.info("agent_service_initialized")

    async def classify_intent(self, query: str) -> str:
        """Classify user intent for routing."""
        response = await self.fast_llm.ainvoke([
            SystemMessage(content=(
                "Classify the user query into one category: "
                "QUESTION, SUMMARIZE, ANALYZE, GENERATE, CODE, OTHER. "
                "Respond with only the category name."
            )),
            HumanMessage(content=query),
        ])
        intent = response.content.strip().upper()
        logger.info("intent_classified", query=query[:80], intent=intent)
        return intent

    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Execute an agent task with optional context."""
        intent = await self.classify_intent(query)

        messages = [
            SystemMessage(content=system_prompt or self._get_system_prompt(intent)),
        ]
        if context:
            messages.append(HumanMessage(
                content=f"Context:\n{context}\n\nTask:\n{query}"
            ))
        else:
            messages.append(HumanMessage(content=query))

        response = await self.llm.ainvoke(messages)

        return {
            "result": response.content,
            "intent": intent,
            "model": self.llm.model_name,
        }

    def _get_system_prompt(self, intent: str) -> str:
        prompts = {
            "QUESTION": "You are a helpful assistant. Answer the question accurately and concisely.",
            "SUMMARIZE": "You are an expert summarizer. Provide clear, structured summaries.",
            "ANALYZE": "You are a data analyst. Provide thorough analysis with evidence-based insights.",
            "GENERATE": "You are a creative writer. Generate high-quality content as requested.",
            "CODE": "You are a senior software engineer. Write clean, well-documented code.",
        }
        return prompts.get(intent, "You are a helpful AI assistant.")
