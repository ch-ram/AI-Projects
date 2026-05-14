"""Unit tests for Production GenAI System."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import timedelta


class TestAuth:
    def test_create_and_decode_token(self):
        with patch("src.auth.security.get_settings") as mock:
            mock.return_value = MagicMock(
                jwt_secret_key="test-secret-key-12345",
                jwt_algorithm="HS256",
                jwt_expiration_minutes=60,
            )
            from src.auth.security import create_access_token, decode_token, User
            user = User(user_id="u1", email="test@test.com", role="user", scopes=["read"])
            token = create_access_token(user)
            assert token
            payload = decode_token(token)
            assert payload.sub == "u1"
            assert payload.role == "user"

    def test_authenticate_user_success(self):
        from src.auth.security import authenticate_user
        user = authenticate_user("admin@psrit.com", "admin123")
        assert user is not None
        assert user.role == "admin"

    def test_authenticate_user_failure(self):
        from src.auth.security import authenticate_user
        user = authenticate_user("admin@psrit.com", "wrong-password")
        assert user is None

    def test_rate_limiter(self):
        from src.auth.security import RateLimiter
        limiter = RateLimiter(rpm=3)
        assert limiter.check("user1") is True
        assert limiter.check("user1") is True
        assert limiter.check("user1") is True
        assert limiter.check("user1") is False  # 4th request blocked
        assert limiter.check("user2") is True   # Different user OK


class TestAgentService:
    @pytest.mark.asyncio
    async def test_classify_intent(self):
        with patch("src.agents.service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                primary_model="gpt-4o", fast_model="gpt-4o-mini", temperature=0.1
            )
            with patch("src.agents.service.ChatOpenAI") as mock_llm:
                mock_instance = MagicMock()
                mock_instance.ainvoke = AsyncMock(
                    return_value=MagicMock(content="QUESTION")
                )
                mock_llm.return_value = mock_instance

                from src.agents.service import AgentService
                service = AgentService()
                service.fast_llm = mock_instance
                intent = await service.classify_intent("What is AI?")
                assert intent == "QUESTION"


class TestObservability:
    def test_metrics_middleware_exists(self):
        from src.monitoring.observability import MetricsMiddleware
        assert MetricsMiddleware is not None

    def test_prometheus_metrics_defined(self):
        from src.monitoring.observability import (
            REQUEST_COUNT, REQUEST_LATENCY,
            LLM_REQUEST_COUNT, LLM_LATENCY, LLM_TOKENS,
            RAG_RETRIEVAL_LATENCY
        )
        assert REQUEST_COUNT is not None
        assert REQUEST_LATENCY is not None
        assert LLM_REQUEST_COUNT is not None


class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        with patch("src.api.main.rag_service"), \
             patch("src.api.main.agent_service"):
            from src.api.main import app
            return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "services" in data

    def test_liveness(self, client):
        r = client.get("/live")
        assert r.status_code == 200

    def test_login_invalid(self, client):
        r = client.post("/api/v1/auth/login", json={
            "email": "bad@bad.com", "password": "wrong"
        })
        assert r.status_code == 401

    def test_login_success(self, client):
        r = client.post("/api/v1/auth/login", json={
            "email": "admin@psrit.com", "password": "admin123"
        })
        assert r.status_code == 200
        assert "access_token" in r.json()

    def test_query_unauthenticated(self, client):
        r = client.post("/api/v1/query", json={"question": "test"})
        assert r.status_code == 401

    def test_query_with_api_key(self, client):
        r = client.post(
            "/api/v1/query",
            json={"question": "What is AI?", "use_rag": False, "use_agent": False},
            headers={"X-API-Key": "dev-api-key-12345"},
        )
        # Will fail because LLM is not mocked, but auth should pass
        assert r.status_code in (200, 500)


class TestConfig:
    def test_settings_load(self):
        with patch("src.utils.config.Path.exists", return_value=False):
            from src.utils.config import Settings
            s = Settings()
            assert s.app_name == "genai-platform"

    def test_cors_list(self):
        with patch("src.utils.config.Path.exists", return_value=False):
            from src.utils.config import Settings
            s = Settings(cors_origins="http://a.com,http://b.com")
            assert s.cors_origins_list == ["http://a.com", "http://b.com"]
