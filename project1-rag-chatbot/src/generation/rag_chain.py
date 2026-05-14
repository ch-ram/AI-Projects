"""
Production RAG chain with conversation memory, streaming, and tracing.
"""
from __future__ import annotations

from typing import AsyncIterator, Optional
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.generation.prompts import (
    get_condense_question_prompt,
    get_rag_prompt,
    get_rag_prompt_with_cot,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """Sliding window conversation memory with optional summarization."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.messages: list[HumanMessage | AIMessage] = []
        self.summary: str = ""

    def add_user_message(self, content: str) -> None:
        self.messages.append(HumanMessage(content=content))
        self._trim()

    def add_ai_message(self, content: str) -> None:
        self.messages.append(AIMessage(content=content))
        self._trim()

    def get_history(self) -> list[HumanMessage | AIMessage]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""

    def _trim(self) -> None:
        if len(self.messages) > self.window_size * 2:
            self.messages = self.messages[-self.window_size * 2 :]


class RAGChain:
    """
    Production RAG chain combining retrieval + generation with:
    - Hybrid retrieval (dense + sparse + re-ranking)
    - Conversation memory (sliding window)
    - Question condensation for multi-turn
    - Streaming token output
    - LangSmith tracing
    """

    def __init__(self, retriever: HybridRetriever):
        settings = get_settings()
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            streaming=True,
        )
        self.sessions: dict[str, ConversationMemory] = {}
        self._output_parser = StrOutputParser()

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing or create new conversation session."""
        sid = session_id or str(uuid4())
        if sid not in self.sessions:
            settings = get_settings()
            self.sessions[sid] = ConversationMemory(
                window_size=settings.memory_window_size
            )
        return sid

    async def aquery(
        self,
        question: str,
        session_id: Optional[str] = None,
        use_cot: bool = False,
    ) -> dict:
        """
        Full RAG query with retrieval, generation, and memory.

        Returns:
            {
                "answer": str,
                "sources": list[dict],
                "session_id": str,
                "retrieval_scores": list[float],
            }
        """
        sid = self.get_or_create_session(session_id)
        memory = self.sessions[sid]

        logger.info("rag_query", session=sid, question=question[:100])

        # Condense question if there's conversation history
        effective_question = question
        if memory.get_history():
            effective_question = await self._condense_question(
                question, memory.get_history()
            )
            logger.debug("condensed_question", original=question, condensed=effective_question)

        # Retrieve
        results = self.retriever.retrieve(effective_question)

        # Format context
        context = self._format_context(results)

        # Generate
        prompt = get_rag_prompt_with_cot() if use_cot else get_rag_prompt()
        chain = prompt | self.llm | self._output_parser

        answer = await chain.ainvoke({
            "context": context,
            "question": question,
            "chat_history": memory.get_history(),
        })

        # Update memory
        memory.add_user_message(question)
        memory.add_ai_message(answer)

        # Build response
        sources = [
            {
                "content": doc.page_content[:300],
                "metadata": doc.metadata,
                "score": round(score, 4),
            }
            for doc, score in results
        ]

        return {
            "answer": answer,
            "sources": sources,
            "session_id": sid,
            "retrieval_scores": [round(s, 4) for _, s in results],
        }

    async def astream(
        self,
        question: str,
        session_id: Optional[str] = None,
        use_cot: bool = False,
    ) -> AsyncIterator[str]:
        """Stream RAG response token by token via SSE."""
        sid = self.get_or_create_session(session_id)
        memory = self.sessions[sid]

        # Condense if multi-turn
        effective_question = question
        if memory.get_history():
            effective_question = await self._condense_question(
                question, memory.get_history()
            )

        # Retrieve
        results = self.retriever.retrieve(effective_question)
        context = self._format_context(results)

        # Stream generate
        prompt = get_rag_prompt_with_cot() if use_cot else get_rag_prompt()
        chain = prompt | self.llm

        full_response = []
        async for chunk in chain.astream({
            "context": context,
            "question": question,
            "chat_history": memory.get_history(),
        }):
            token = chunk.content
            if token:
                full_response.append(token)
                yield token

        # Update memory
        complete_answer = "".join(full_response)
        memory.add_user_message(question)
        memory.add_ai_message(complete_answer)

    async def _condense_question(
        self,
        question: str,
        chat_history: list,
    ) -> str:
        """Condense a follow-up question into a standalone question."""
        prompt = get_condense_question_prompt()
        chain = prompt | self.llm | self._output_parser

        history_text = "\n".join(
            f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in chat_history[-6:]  # Last 3 exchanges
        )

        return await chain.ainvoke({
            "chat_history": history_text,
            "question": question,
        })

    @staticmethod
    def _format_context(results: list[tuple[Document, float]]) -> str:
        """Format retrieved documents into context string."""
        if not results:
            return "No relevant documents found."

        parts = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("file_name", "Unknown")
            parts.append(
                f"[Document {i}] (Source: {source}, Relevance: {score:.3f})\n"
                f"{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    def clear_session(self, session_id: str) -> bool:
        """Clear a conversation session."""
        if session_id in self.sessions:
            self.sessions[session_id].clear()
            del self.sessions[session_id]
            return True
        return False
