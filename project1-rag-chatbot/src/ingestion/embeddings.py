"""
Embedding generation and FAISS index management.
Handles batch embedding, index creation, persistence, and deduplication.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingManager:
    """Manages OpenAI embeddings and FAISS vector index."""

    def __init__(self):
        settings = get_settings()
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dimensions,
        )
        self.dimension = settings.openai_embedding_dimensions
        self.index_path = Path(settings.faiss_index_path)
        self.index: Optional[faiss.Index] = None
        self.documents: list[Document] = []
        self._doc_hashes: set[str] = set()

    def create_index(self, documents: list[Document]) -> faiss.Index:
        """Create a FAISS index from documents."""
        settings = get_settings()

        # Deduplicate by content hash
        unique_docs = []
        for doc in documents:
            content_hash = doc.metadata.get("content_hash", "")
            if content_hash and content_hash in self._doc_hashes:
                logger.debug("skipping_duplicate", hash=content_hash)
                continue
            self._doc_hashes.add(content_hash)
            unique_docs.append(doc)

        logger.info(
            "creating_embeddings",
            total=len(documents),
            unique=len(unique_docs),
        )

        # Batch embed
        texts = [doc.page_content for doc in unique_docs]
        vectors = self._batch_embed(texts)
        vectors_np = np.array(vectors, dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(vectors_np)

        # Build index based on dataset size
        n_vectors = len(vectors_np)
        if n_vectors < 1000:
            # Small dataset: flat index
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            # Large dataset: IVFFlat
            nlist = min(settings.faiss_nlist, n_vectors // 10)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(vectors_np)
            self.index.nprobe = settings.faiss_nprobe

        self.index.add(vectors_np)
        self.documents = unique_docs

        logger.info("index_created", vectors=n_vectors, dimension=self.dimension)
        return self.index

    def add_documents(self, documents: list[Document]) -> int:
        """Add documents to existing index."""
        if self.index is None:
            return len(self.create_index(documents).ntotal)

        unique_docs = [
            d for d in documents
            if d.metadata.get("content_hash", "") not in self._doc_hashes
        ]
        if not unique_docs:
            return 0

        texts = [doc.page_content for doc in unique_docs]
        vectors = self._batch_embed(texts)
        vectors_np = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors_np)

        self.index.add(vectors_np)
        self.documents.extend(unique_docs)
        for doc in unique_docs:
            self._doc_hashes.add(doc.metadata.get("content_hash", ""))

        logger.info("documents_added", count=len(unique_docs))
        return len(unique_docs)

    def search(
        self, query: str, top_k: int = 5
    ) -> list[tuple[Document, float]]:
        """Search the FAISS index and return (document, score) pairs."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("empty_index_search")
            return []

        query_vector = np.array(
            [self.embeddings.embed_query(query)], dtype=np.float32
        )
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            results.append((self.documents[idx], float(score)))

        return results

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save")

        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))

        # Save document store
        import json
        docs_data = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in self.documents
        ]
        with open(self.index_path / "documents.json", "w") as f:
            json.dump(docs_data, f, indent=2, default=str)

        logger.info("index_saved", path=str(self.index_path))

    def load(self) -> bool:
        """Load index and metadata from disk."""
        index_file = self.index_path / "index.faiss"
        docs_file = self.index_path / "documents.json"

        if not index_file.exists() or not docs_file.exists():
            logger.warning("no_persisted_index", path=str(self.index_path))
            return False

        self.index = faiss.read_index(str(index_file))

        import json
        with open(docs_file) as f:
            docs_data = json.load(f)

        self.documents = [
            Document(page_content=d["content"], metadata=d["metadata"])
            for d in docs_data
        ]
        self._doc_hashes = {
            d["metadata"].get("content_hash", "") for d in docs_data
        }

        logger.info(
            "index_loaded",
            vectors=self.index.ntotal,
            documents=len(self.documents),
        )
        return True

    def _batch_embed(
        self, texts: list[str], batch_size: int = 100
    ) -> list[list[float]]:
        """Batch embedding with progress tracking."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.embeddings.embed_documents(batch)
            all_embeddings.extend(embeddings)
            logger.debug(
                "embedding_batch",
                progress=f"{min(i + batch_size, len(texts))}/{len(texts)}",
            )
        return all_embeddings
