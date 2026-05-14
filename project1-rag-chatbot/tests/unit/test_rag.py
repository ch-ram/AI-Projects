"""
Unit tests for RAG Chatbot components.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

from langchain_core.documents import Document


# ── Document Loader Tests ──────────────────────────────────

class TestDocumentLoader:
    """Tests for multi-format document loading."""

    def test_supported_extensions(self):
        from src.ingestion.loaders import DocumentLoader
        loader = DocumentLoader()
        assert ".pdf" in loader.SUPPORTED_EXTENSIONS
        assert ".docx" in loader.SUPPORTED_EXTENSIONS
        assert ".txt" in loader.SUPPORTED_EXTENSIONS
        assert ".md" in loader.SUPPORTED_EXTENSIONS
        assert ".html" in loader.SUPPORTED_EXTENSIONS

    def test_unsupported_extension_raises(self, tmp_path):
        from src.ingestion.loaders import DocumentLoader
        loader = DocumentLoader()
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("test")
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load_file(bad_file)

    def test_missing_file_raises(self):
        from src.ingestion.loaders import DocumentLoader
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/file.txt")

    def test_load_text_file(self, tmp_path):
        from src.ingestion.loaders import DocumentLoader
        loader = DocumentLoader()
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello World. This is a test document.")
        docs = loader.load_file(txt_file)
        assert len(docs) == 1
        assert "Hello World" in docs[0].page_content
        assert docs[0].metadata["file_type"] == ".txt"
        assert docs[0].metadata["content_hash"]

    def test_load_directory(self, tmp_path):
        from src.ingestion.loaders import DocumentLoader
        loader = DocumentLoader()
        (tmp_path / "a.txt").write_text("Document A")
        (tmp_path / "b.txt").write_text("Document B")
        (tmp_path / "c.xyz").write_text("Ignored file")
        docs = loader.load_directory(tmp_path)
        assert len(docs) == 2


# ── Chunker Tests ──────────────────────────────────────────

class TestChunkers:
    """Tests for document chunking strategies."""

    def test_recursive_chunker(self):
        from src.ingestion.chunkers import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        doc = Document(page_content="A" * 200, metadata={"source": "test"})
        chunks = chunker.split([doc])
        assert len(chunks) > 1
        assert all(c.metadata.get("chunk_index") is not None for c in chunks)

    def test_chunker_factory(self):
        from src.ingestion.chunkers import get_chunker, RecursiveChunker
        with patch("src.ingestion.chunkers.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                chunking_strategy="recursive",
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunker = get_chunker("recursive")
            assert isinstance(chunker, RecursiveChunker)


# ── Retriever Tests ────────────────────────────────────────

class TestHybridRetriever:
    """Tests for hybrid retrieval engine."""

    def test_reciprocal_rank_fusion(self):
        from src.retrieval.hybrid_retriever import HybridRetriever

        doc_a = Document(page_content="A", metadata={"content_hash": "a"})
        doc_b = Document(page_content="B", metadata={"content_hash": "b"})
        doc_c = Document(page_content="C", metadata={"content_hash": "c"})

        list1 = [(doc_a, 0.9), (doc_b, 0.7)]
        list2 = [(doc_b, 0.8), (doc_c, 0.6)]

        fused = HybridRetriever._reciprocal_rank_fusion(
            list1, list2, top_k=3
        )
        assert len(fused) <= 3
        # doc_b should rank highest (appears in both lists)
        assert fused[0][0].page_content == "B"

    def test_bm25_retriever(self):
        from src.retrieval.hybrid_retriever import BM25Retriever
        docs = [
            Document(page_content="Python programming language basics"),
            Document(page_content="Java enterprise application development"),
            Document(page_content="Python machine learning with scikit-learn"),
        ]
        bm25 = BM25Retriever(docs)
        results = bm25.search("Python machine learning", top_k=2)
        assert len(results) <= 2
        assert "Python" in results[0][0].page_content


# ── Memory Tests ───────────────────────────────────────────

class TestConversationMemory:
    """Tests for conversation memory management."""

    def test_sliding_window(self):
        from src.generation.rag_chain import ConversationMemory
        memory = ConversationMemory(window_size=2)
        for i in range(10):
            memory.add_user_message(f"Q{i}")
            memory.add_ai_message(f"A{i}")
        history = memory.get_history()
        assert len(history) <= 4  # 2 exchanges = 4 messages

    def test_clear(self):
        from src.generation.rag_chain import ConversationMemory
        memory = ConversationMemory()
        memory.add_user_message("Hello")
        memory.clear()
        assert len(memory.get_history()) == 0


# ── API Tests ──────────────────────────────────────────────

class TestAPI:
    """Tests for FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        with patch("src.api.main.embedding_manager"), \
             patch("src.api.main.rag_chain"):
            from src.api.main import app
            return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_query_requires_body(self, client):
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422  # Validation error
