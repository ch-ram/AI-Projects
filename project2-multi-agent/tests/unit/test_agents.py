"""Unit tests for Multi-Agent Research Assistant."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestAgentDefinitions:
    def test_all_agents_defined(self):
        from src.agents.definitions import ALL_AGENTS
        assert len(ALL_AGENTS) == 4
        names = {a.name for a in ALL_AGENTS}
        assert names == {"research_agent", "analysis_agent", "writer_agent", "reviewer_agent"}

    def test_agent_has_required_fields(self):
        from src.agents.definitions import RESEARCH_AGENT
        assert RESEARCH_AGENT.name
        assert RESEARCH_AGENT.role
        assert RESEARCH_AGENT.goal
        assert RESEARCH_AGENT.backstory
        assert len(RESEARCH_AGENT.tools) > 0


class TestToolRegistry:
    def test_registry_has_all_tools(self):
        from src.tools.registry import TOOL_REGISTRY
        expected = {"web_search", "arxiv_search", "wikipedia_search", "calculator", "code_executor", "markdown_formatter", "memory_lookup"}
        assert set(TOOL_REGISTRY.keys()) == expected

    def test_calculator_tool(self):
        from src.tools.registry import calculator
        result = calculator.invoke("2 + 3 * 4")
        assert "14" in result

    def test_get_tools_for_agent(self):
        from src.tools.registry import get_tools_for_agent
        tools = get_tools_for_agent(["web_search", "calculator"])
        assert len(tools) == 2


class TestSharedMemory:
    def test_in_memory_store_and_search(self):
        with patch("src.memory.shared_memory.get_settings") as mock:
            mock.return_value = MagicMock(vector_db_provider="in_memory")
            from src.memory.shared_memory import SharedMemory
            memory = SharedMemory()
            memory.store("Python is a programming language", "research_agent")
            memory.store("Java is used for enterprise apps", "research_agent")
            results = memory.search("Python programming")
            assert len(results) > 0


class TestOrchestrator:
    def test_graph_builds(self):
        from src.orchestrator.graph import build_research_graph
        graph = build_research_graph()
        assert graph is not None

    def test_should_revise_logic(self):
        from src.orchestrator.graph import should_revise
        state_approve = {"status": "approved", "revision_count": 1, "max_revisions": 2, "error": None}
        assert should_revise(state_approve) == "finalize"
        state_revise = {"status": "needs_revision", "revision_count": 0, "max_revisions": 2, "error": None}
        assert should_revise(state_revise) == "revise"
        state_max = {"status": "needs_revision", "revision_count": 2, "max_revisions": 2, "error": None}
        assert should_revise(state_max) == "finalize"


class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        with patch("src.api.main.orchestrator"):
            from src.api.main import app
            return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert "agents" in r.json()
