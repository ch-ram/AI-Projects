"""
Production RAG service with multi-provider vector store support.
Supports Pinecone, pgvector, and FAISS backends.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


RAG_SYSTEM_PROMPT = """You are an expert AI assistant for the GenAI Platform.
Answer questions based ONLY on the provided context documents.

Context Documents:
{context}

Rules:
- Only use information from the provided context
- Cite document sources when possible
- Say "I don't have enough information" if the context doesn't cover the question
- Be concise and accurate"""


class VectorStoreFactory:
    """Create vector store instances based on configuration."""

    @staticmethod
    def create(provider: Optional[str] = None):
        settings = get_settings()
        provider = provider or settings.vector_db_provider

        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

        if provider == "pinecone":
            from langchain_pinecone import PineconeVectorStore
            return PineconeVectorStore(
                index_name=settings.pinecone_index,
                embedding=embeddings,
                pinecone_api_key=settings.pinecone_api_key,
            )
        elif provider == "pgvector":
            from langchain_community.vectorstores import PGVector
            return PGVector(
                connection_string=settings.pgvector_url,
                embedding_function=embeddings,
                collection_name="documents",
            )
        elif provider == "faiss":
            from langchain_community.vectorstores import FAISS
            # Load or create FAISS index
            try:
                return FAISS.load_local(
                    "./data/faiss_index",
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                logger.warning("faiss_index_not_found_creating_empty")
                import faiss
                import numpy as np
                index = faiss.IndexFlatIP(settings.embedding_dimensions)
                return FAISS(
                    embedding_function=embeddings,
                    index=index,
                    docstore={},
                    index_to_docstore_id={},
                )
        else:
            raise ValueError(f"Unknown vector store provider: {provider}")


class RAGService:
    """
    Production RAG service with:
    - Multi-provider vector store
    - Streaming generation
    - Source tracking
    - Conversation context
    """

    def __init__(self):
        settings = get_settings()
        self.vector_store = VectorStoreFactory.create()
        self.llm = ChatOpenAI(
            model=settings.primary_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            streaming=True,
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": settings.retrieval_top_k}
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()
        logger.info("rag_service_initialized", provider=settings.vector_db_provider)

    async def query(
        self,
        question: str,
        chat_history: Optional[list] = None,
    ) -> dict:
        """Execute RAG query with retrieval and generation."""
        # Retrieve
        docs = await self.retriever.ainvoke(question)
        context = self._format_context(docs)

        # Generate
        answer = await self.chain.ainvoke({
            "context": context,
            "question": question,
            "chat_history": chat_history or [],
        })

        sources = [
            {
                "content": d.page_content[:300],
                "metadata": d.metadata,
            }
            for d in docs
        ]

        return {"answer": answer, "sources": sources}

    async def ingest(self, documents: list[Document]) -> int:
        """Add documents to the vector store."""
        if hasattr(self.vector_store, "aadd_documents"):
            ids = await self.vector_store.aadd_documents(documents)
        else:
            ids = self.vector_store.add_documents(documents)
        logger.info("documents_ingested", count=len(ids))
        return len(ids)

    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        if not docs:
            return "No relevant documents found."
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            parts.append(f"[Doc {i}] (Source: {source})\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)
