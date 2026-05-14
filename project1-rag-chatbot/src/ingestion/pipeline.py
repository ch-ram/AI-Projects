"""
End-to-end document ingestion pipeline.
Orchestrates: Load → Chunk → Embed → Index → Persist
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src.ingestion.loaders import DocumentLoader
from src.ingestion.chunkers import get_chunker
from src.ingestion.embeddings import EmbeddingManager
from src.utils.config import get_settings
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrates the full document ingestion workflow."""

    def __init__(
        self,
        chunking_strategy: Optional[str] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        self.loader = DocumentLoader()
        self.chunker = get_chunker(chunking_strategy)
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self._stats = {
            "files_processed": 0,
            "files_failed": 0,
            "raw_documents": 0,
            "chunks_created": 0,
            "vectors_indexed": 0,
        }

    def ingest_file(self, file_path: str | Path) -> int:
        """Ingest a single file into the vector store."""
        try:
            docs = self.loader.load_file(file_path)
            self._stats["raw_documents"] += len(docs)

            chunks = self.chunker.split(docs)
            self._stats["chunks_created"] += len(chunks)

            added = self.embedding_manager.add_documents(chunks)
            self._stats["vectors_indexed"] += added
            self._stats["files_processed"] += 1

            logger.info(
                "file_ingested",
                file=str(file_path),
                raw_docs=len(docs),
                chunks=len(chunks),
                indexed=added,
            )
            return added
        except Exception as e:
            self._stats["files_failed"] += 1
            logger.error("ingestion_failed", file=str(file_path), error=str(e))
            raise

    def ingest_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
        persist: bool = True,
    ) -> dict:
        """Ingest all documents from a directory."""
        logger.info("pipeline_start", directory=str(dir_path))

        docs = self.loader.load_directory(dir_path, recursive=recursive)
        self._stats["raw_documents"] = len(docs)

        chunks = self.chunker.split(docs)
        self._stats["chunks_created"] = len(chunks)

        self.embedding_manager.create_index(chunks)
        self._stats["vectors_indexed"] = self.embedding_manager.index.ntotal

        if persist:
            self.embedding_manager.save()

        self._stats["files_processed"] = len(
            set(d.metadata.get("source", "") for d in docs)
        )

        logger.info("pipeline_complete", stats=self._stats)
        return self._stats

    @property
    def stats(self) -> dict:
        return self._stats.copy()


def main():
    """CLI entry point for ingestion pipeline."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Document Ingestion Pipeline")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["recursive", "semantic", "sentence"],
        default=None,
        help="Chunking strategy (default: from config)",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not persist index to disk",
    )
    args = parser.parse_args()

    pipeline = IngestionPipeline(chunking_strategy=args.strategy)
    stats = pipeline.ingest_directory(
        args.input_dir,
        persist=not args.no_persist,
    )

    print(f"\n{'='*50}")
    print("Ingestion Complete")
    print(f"{'='*50}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
