"""
Multi-format document loading pipeline.
Supports PDF, DOCX, TXT, Markdown, HTML with metadata extraction.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Unified document loader supporting multiple file formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}

    def load_file(self, file_path: str | Path) -> list[Document]:
        """Load a single file and return LangChain Documents with metadata."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        logger.info("loading_document", file=str(path), type=ext)

        loader_map = {
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".txt": self._load_text,
            ".md": self._load_markdown,
            ".html": self._load_html,
            ".htm": self._load_html,
        }

        docs = loader_map[ext](path)

        # Enrich metadata
        for doc in docs:
            doc.metadata.update({
                "source": str(path),
                "file_name": path.name,
                "file_type": ext,
                "file_size_bytes": path.stat().st_size,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "content_hash": hashlib.sha256(
                    doc.page_content.encode()
                ).hexdigest()[:16],
            })

        logger.info("document_loaded", file=str(path), chunks=len(docs))
        return docs

    def load_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
        extensions: Optional[set[str]] = None,
    ) -> list[Document]:
        """Load all supported documents from a directory."""
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        allowed = extensions or self.SUPPORTED_EXTENSIONS
        pattern = "**/*" if recursive else "*"
        files = [
            f for f in path.glob(pattern)
            if f.is_file() and f.suffix.lower() in allowed
        ]

        logger.info("loading_directory", dir=str(path), file_count=len(files))

        all_docs = []
        for file in sorted(files):
            try:
                docs = self.load_file(file)
                all_docs.extend(docs)
            except Exception as e:
                logger.error("file_load_error", file=str(file), error=str(e))

        return all_docs

    def _load_pdf(self, path: Path) -> list[Document]:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(path))
        return loader.load()

    def _load_docx(self, path: Path) -> list[Document]:
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(str(path))
        return loader.load()

    def _load_text(self, path: Path) -> list[Document]:
        text = path.read_text(encoding="utf-8")
        return [Document(page_content=text)]

    def _load_markdown(self, path: Path) -> list[Document]:
        from langchain_community.document_loaders import UnstructuredMarkdownLoader
        loader = UnstructuredMarkdownLoader(str(path))
        return loader.load()

    def _load_html(self, path: Path) -> list[Document]:
        from langchain_community.document_loaders import BSHTMLLoader
        loader = BSHTMLLoader(str(path))
        return loader.load()
