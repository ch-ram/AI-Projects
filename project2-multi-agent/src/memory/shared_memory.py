"""
Shared memory store for multi-agent collaboration.
Agents can read/write to shared memory to coordinate findings.
Supports ChromaDB (dev) and Pinecone (prod).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SharedMemory:
    """
    Shared memory for multi-agent collaboration.
    Stores agent outputs, research findings, and cross-references.
    """

    def __init__(self):
        settings = get_settings()
        self.provider = settings.vector_db_provider

        if self.provider == "chromadb":
            self._init_chromadb()
        elif self.provider == "pinecone":
            self._init_pinecone()
        else:
            self._init_in_memory()

    def _init_chromadb(self):
        import chromadb
        settings = get_settings()
        self.client = chromadb.PersistentClient(path=settings.chromadb_path)
        self.collection = self.client.get_or_create_collection(
            name="agent_memory",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("shared_memory_chromadb", path=settings.chromadb_path)

    def _init_pinecone(self):
        from pinecone import Pinecone
        settings = get_settings()
        pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = pc.Index(settings.pinecone_index)
        logger.info("shared_memory_pinecone", index=settings.pinecone_index)

    def _init_in_memory(self):
        self._store: list[dict] = []
        logger.info("shared_memory_in_memory")

    def store(
        self,
        content: str,
        agent_name: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store agent output in shared memory."""
        doc_id = str(uuid4())
        meta = {
            "agent": agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }

        if self.provider == "chromadb":
            self.collection.add(
                documents=[content],
                ids=[doc_id],
                metadatas=[meta],
            )
        elif self.provider == "pinecone":
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings()
            vector = embeddings.embed_query(content)
            self.index.upsert(
                vectors=[{
                    "id": doc_id,
                    "values": vector,
                    "metadata": {**meta, "content": content[:1000]},
                }]
            )
        else:
            self._store.append({"id": doc_id, "content": content, **meta})

        logger.debug("memory_stored", id=doc_id, agent=agent_name)
        return doc_id

    def search(
        self, query: str, top_k: int = 5, agent_filter: Optional[str] = None
    ) -> list[dict]:
        """Search shared memory for relevant information."""
        if self.provider == "chromadb":
            where = {"agent": agent_filter} if agent_filter else None
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
            )
            return [
                {
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                }
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]
        elif self.provider == "pinecone":
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings()
            vector = embeddings.embed_query(query)
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
            )
            return [
                {
                    "content": m.metadata.get("content", ""),
                    "metadata": m.metadata,
                    "score": m.score,
                }
                for m in results.matches
            ]
        else:
            # Simple keyword matching for in-memory
            matched = []
            query_terms = set(query.lower().split())
            for item in self._store:
                content_terms = set(item["content"].lower().split())
                overlap = len(query_terms & content_terms)
                if overlap > 0:
                    matched.append(({**item, "score": overlap}))
            matched.sort(key=lambda x: x["score"], reverse=True)
            return matched[:top_k]

    def clear(self) -> None:
        """Clear all shared memory."""
        if self.provider == "chromadb":
            self.client.delete_collection("agent_memory")
            self.collection = self.client.get_or_create_collection(
                name="agent_memory"
            )
        elif self.provider == "pinecone":
            self.index.delete(delete_all=True)
        else:
            self._store.clear()
        logger.info("memory_cleared")
