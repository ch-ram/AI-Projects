"""
Document chunking strategies with configurable splitting.
Supports Recursive, Semantic, and Sentence-based splitting.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"


class BaseChunker(ABC):
    """Abstract base for chunking strategies."""

    @abstractmethod
    def split(self, documents: list[Document]) -> list[Document]:
        ...


class RecursiveChunker(BaseChunker):
    """Recursive character text splitter with overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        chunks = self.splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        return chunks


class SemanticChunker(BaseChunker):
    """Semantic chunking using embedding similarity breakpoints."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.fallback = RecursiveChunker(chunk_size, chunk_overlap)

    def split(self, documents: list[Document]) -> list[Document]:
        try:
            from langchain_experimental.text_splitter import SemanticChunker as LC_Semantic
            from langchain_openai import OpenAIEmbeddings

            settings = get_settings()
            embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                dimensions=settings.openai_embedding_dimensions,
            )
            splitter = LC_Semantic(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90,
            )
            chunks = splitter.split_documents(documents)
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_size"] = len(chunk.page_content)
                chunk.metadata["chunking"] = "semantic"
            return chunks
        except ImportError:
            logger.warning("semantic_chunker_unavailable_falling_back_to_recursive")
            return self.fallback.split(documents)


class SentenceChunker(BaseChunker):
    """Sentence-transformer token-aware chunking."""

    def __init__(self, chunk_overlap: int = 50, tokens_per_chunk: int = 256):
        self.splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        chunks = self.splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            chunk.metadata["chunking"] = "sentence"
        return chunks


def get_chunker(strategy: str | None = None) -> BaseChunker:
    """Factory for chunking strategy."""
    settings = get_settings()
    strategy = strategy or settings.chunking_strategy

    chunker_map = {
        ChunkingStrategy.RECURSIVE: lambda: RecursiveChunker(
            settings.chunk_size, settings.chunk_overlap
        ),
        ChunkingStrategy.SEMANTIC: lambda: SemanticChunker(
            settings.chunk_size, settings.chunk_overlap
        ),
        ChunkingStrategy.SENTENCE: lambda: SentenceChunker(),
    }

    factory = chunker_map.get(ChunkingStrategy(strategy))
    if not factory:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    logger.info("chunker_initialized", strategy=strategy)
    return factory()
