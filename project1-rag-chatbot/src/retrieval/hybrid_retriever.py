"""
Hybrid retrieval engine combining Dense (FAISS) + Sparse (BM25) + Re-ranking.
Supports multiple retrieval strategies with configurable fusion.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from langchain_core.documents import Document

from src.ingestion.embeddings import EmbeddingManager
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RetrievalStrategy(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class BM25Retriever:
    """Sparse retrieval using BM25 scoring."""

    def __init__(self, documents: list[Document]):
        from rank_bm25 import BM25Okapi
        self.documents = documents
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("bm25_initialized", doc_count=len(documents))

    def search(self, query: str, top_k: int = 10) -> list[tuple[Document, float]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))
        return results


class CrossEncoderReranker:
    """Re-rank results using a cross-encoder model."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        try:
            from flashrank import Ranker, RerankRequest
            self.ranker = Ranker(model_name=model_name)
            self._use_flashrank = True
            logger.info("reranker_initialized", model=model_name, backend="flashrank")
        except ImportError:
            self._use_flashrank = False
            logger.warning("flashrank_unavailable_reranking_disabled")

    def rerank(
        self,
        query: str,
        documents: list[tuple[Document, float]],
        top_k: int = 3,
    ) -> list[tuple[Document, float]]:
        if not self._use_flashrank or not documents:
            return documents[:top_k]

        from flashrank import Ranker, RerankRequest

        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, (doc, _) in enumerate(documents)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)

        reranked = []
        for result in results[:top_k]:
            idx = result["id"]
            doc = documents[idx][0]
            reranked.append((doc, result["score"]))

        logger.debug("reranked", input=len(documents), output=len(reranked))
        return reranked


class HybridRetriever:
    """
    Production hybrid retriever combining dense and sparse search.

    Supports:
    - Dense-only (FAISS cosine similarity)
    - Sparse-only (BM25)
    - Hybrid (Reciprocal Rank Fusion of dense + sparse)
    - Optional cross-encoder re-ranking
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        strategy: Optional[str] = None,
    ):
        settings = get_settings()
        self.embedding_manager = embedding_manager
        self.strategy = RetrievalStrategy(strategy or settings.retrieval_strategy)
        self.bm25_weight = settings.bm25_weight
        self.dense_weight = settings.dense_weight
        self.top_k = settings.retrieval_top_k

        # Initialize BM25 if needed
        self.bm25: Optional[BM25Retriever] = None
        if self.strategy in (RetrievalStrategy.SPARSE, RetrievalStrategy.HYBRID):
            if embedding_manager.documents:
                self.bm25 = BM25Retriever(embedding_manager.documents)

        # Initialize re-ranker if enabled
        self.reranker: Optional[CrossEncoderReranker] = None
        if settings.reranker_enabled:
            self.reranker = CrossEncoderReranker(settings.reranker_model)
            self.reranker_top_k = settings.reranker_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results to return
            strategy: Override retrieval strategy

        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        k = top_k or self.top_k
        strat = RetrievalStrategy(strategy) if strategy else self.strategy

        logger.info("retrieving", query=query[:100], strategy=strat, top_k=k)

        if strat == RetrievalStrategy.DENSE:
            results = self._dense_search(query, k)
        elif strat == RetrievalStrategy.SPARSE:
            results = self._sparse_search(query, k)
        elif strat == RetrievalStrategy.HYBRID:
            results = self._hybrid_search(query, k)
        else:
            results = self._dense_search(query, k)

        # Apply re-ranking
        if self.reranker and results:
            results = self.reranker.rerank(
                query, results, self.reranker_top_k
            )

        logger.info("retrieval_complete", results=len(results))
        return results

    def _dense_search(
        self, query: str, top_k: int
    ) -> list[tuple[Document, float]]:
        """FAISS cosine similarity search."""
        return self.embedding_manager.search(query, top_k)

    def _sparse_search(
        self, query: str, top_k: int
    ) -> list[tuple[Document, float]]:
        """BM25 keyword search."""
        if self.bm25 is None:
            logger.warning("bm25_not_initialized_falling_back_to_dense")
            return self._dense_search(query, top_k)
        return self.bm25.search(query, top_k)

    def _hybrid_search(
        self, query: str, top_k: int
    ) -> list[tuple[Document, float]]:
        """Reciprocal Rank Fusion of dense + sparse results."""
        fetch_k = top_k * 3  # Over-fetch for fusion

        dense_results = self._dense_search(query, fetch_k)
        sparse_results = self._sparse_search(query, fetch_k)

        return self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            top_k=top_k,
            weights=(self.dense_weight, self.bm25_weight),
        )

    @staticmethod
    def _reciprocal_rank_fusion(
        *result_lists: list[tuple[Document, float]],
        top_k: int = 5,
        weights: tuple[float, ...] = (1.0, 1.0),
        k: int = 60,
    ) -> list[tuple[Document, float]]:
        """
        Reciprocal Rank Fusion (RRF) to combine multiple ranked lists.
        Score = sum(weight_i / (k + rank_i)) for each list.
        """
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for weight, results in zip(weights, result_lists):
            for rank, (doc, _score) in enumerate(results):
                doc_id = doc.metadata.get("content_hash", str(id(doc)))
                rrf_score = weight / (k + rank + 1)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
                doc_map[doc_id] = doc

        sorted_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)

        return [
            (doc_map[doc_id], doc_scores[doc_id])
            for doc_id in sorted_ids[:top_k]
        ]
