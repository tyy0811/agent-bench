"""Tests for conversation memory store."""

from __future__ import annotations

import pytest

from agent_bench.memory.store import ConversationStore


@pytest.fixture
def store(tmp_path) -> ConversationStore:
    """ConversationStore with a temp DB path."""
    return ConversationStore(db_path=str(tmp_path / "test.db"))


class TestConversationStore:
    def test_append_and_retrieve(self, store: ConversationStore):
        """Write 3 messages, read back in chronological order."""
        store.append("s1", "user", "Hello")
        store.append("s1", "assistant", "Hi there")
        store.append("s1", "user", "How are you?")

        history = store.get_history("s1")
        assert len(history) == 3
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there"}
        assert history[2] == {"role": "user", "content": "How are you?"}

    def test_max_turns(self, store: ConversationStore):
        """max_turns=2 returns at most 4 messages (2 user + 2 assistant)."""
        for i in range(10):
            store.append("s1", "user", f"Q{i}")
            store.append("s1", "assistant", f"A{i}")

        history = store.get_history("s1", max_turns=2)
        assert len(history) == 4  # 2 turns * 2 messages each

    def test_separate_sessions(self, store: ConversationStore):
        """Two session_ids don't cross-contaminate."""
        store.append("s1", "user", "Session 1 message")
        store.append("s2", "user", "Session 2 message")

        h1 = store.get_history("s1")
        h2 = store.get_history("s2")

        assert len(h1) == 1
        assert len(h2) == 1
        assert h1[0]["content"] == "Session 1 message"
        assert h2[0]["content"] == "Session 2 message"

    def test_empty_session(self, store: ConversationStore):
        """Non-existent session returns empty list."""
        assert store.get_history("nonexistent") == []

    def test_list_sessions(self, store: ConversationStore):
        """List all session IDs."""
        store.append("alpha", "user", "msg")
        store.append("beta", "user", "msg")
        store.append("alpha", "user", "msg2")

        sessions = store.list_sessions()
        assert set(sessions) == {"alpha", "beta"}

    def test_delete_session(self, store: ConversationStore):
        """Delete removes all messages for a session."""
        store.append("s1", "user", "keep")
        store.append("s2", "user", "delete me")

        store.delete_session("s2")

        assert store.get_history("s1") == [{"role": "user", "content": "keep"}]
        assert store.get_history("s2") == []

    def test_metadata_stored(self, store: ConversationStore):
        """Metadata is accepted without error (not exposed in get_history)."""
        store.append("s1", "user", "test", metadata={"sources": ["doc.md"]})
        history = store.get_history("s1")
        assert len(history) == 1


def _make_session_app(tmp_path):
    """Create a test app WITH conversation store attached."""
    import time as time_mod

    from fastapi import FastAPI

    from agent_bench.agents.orchestrator import Orchestrator
    from agent_bench.core.config import AppConfig, MemoryConfig, ProviderConfig
    from agent_bench.core.provider import MockProvider
    from agent_bench.memory.store import ConversationStore
    from agent_bench.rag.store import HybridStore
    from agent_bench.serving.middleware import MetricsCollector, RequestMiddleware
    from agent_bench.tools.calculator import CalculatorTool
    from agent_bench.tools.registry import ToolRegistry
    from tests.test_agent import FakeSearchTool

    app = FastAPI(title="agent-bench-session-test")

    registry = ToolRegistry()
    registry.register(FakeSearchTool())
    registry.register(CalculatorTool())

    provider = MockProvider()
    orchestrator = Orchestrator(
        provider=provider, registry=registry, max_iterations=3
    )

    config = AppConfig(
        provider=ProviderConfig(default="mock"),
        memory=MemoryConfig(
            enabled=True,
            db_path=str(tmp_path / "test_sessions.db"),
            max_turns=10,
        ),
    )
    conversation_store = ConversationStore(
        db_path=config.memory.db_path
    )

    app.state.orchestrator = orchestrator
    app.state.store = HybridStore(dimension=384)
    app.state.conversation_store = conversation_store
    app.state.config = config
    app.state.system_prompt = "You are a test assistant."
    app.state.start_time = time_mod.time()
    app.state.metrics = MetricsCollector()

    app.add_middleware(RequestMiddleware)

    from agent_bench.serving.routes import router

    app.include_router(router)
    return app, conversation_store


class TestSessionIntegration:
    @pytest.mark.asyncio
    async def test_stateless_without_session_id(self, tmp_path):
        """session_id=None suppresses DB interaction even when store exists."""
        from httpx import ASGITransport, AsyncClient

        app, conv_store = _make_session_app(tmp_path)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/ask", json={"question": "test"}
            )
        assert response.status_code == 200
        assert "answer" in response.json()
        # No session_id → nothing stored
        assert conv_store.list_sessions() == []

    @pytest.mark.asyncio
    async def test_session_stores_and_loads_history(self, tmp_path):
        """Two requests with same session_id: second uses stored history."""
        from httpx import ASGITransport, AsyncClient

        app, conv_store = _make_session_app(tmp_path)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # First request with session_id
            r1 = await client.post(
                "/ask",
                json={"question": "What is FastAPI?", "session_id": "sess-1"},
            )
            assert r1.status_code == 200

            # Verify Q+A was stored
            history = conv_store.get_history("sess-1")
            assert len(history) == 2
            assert history[0]["role"] == "user"
            assert history[0]["content"] == "What is FastAPI?"
            assert history[1]["role"] == "assistant"

            # Second request in same session
            r2 = await client.post(
                "/ask",
                json={
                    "question": "Tell me more about it",
                    "session_id": "sess-1",
                },
            )
            assert r2.status_code == 200

            # Now 4 messages stored (2 turns)
            history = conv_store.get_history("sess-1")
            assert len(history) == 4

    @pytest.mark.asyncio
    async def test_history_passed_to_orchestrator(self, tmp_path):
        """Verify the orchestrator actually receives history on follow-up."""
        from httpx import ASGITransport, AsyncClient

        from agent_bench.agents.orchestrator import AgentResponse
        from agent_bench.core.types import TokenUsage

        app, conv_store = _make_session_app(tmp_path)

        # Seed a prior conversation turn in the store
        conv_store.append("sess-2", "user", "What is FastAPI?")
        conv_store.append("sess-2", "assistant", "FastAPI is a web framework.")

        # Patch orchestrator.run to capture the history argument
        captured_kwargs: dict = {}
        fake_response = AgentResponse(
            answer="Follow-up answer.",
            sources=[],
            iterations=1,
            tools_used=[],
            usage=TokenUsage(
                input_tokens=100,
                output_tokens=20,
                estimated_cost_usd=0.0001,
            ),
            provider="mock",
            model="mock-1",
            latency_ms=1.0,
        )

        async def spy_run(**kwargs):
            captured_kwargs.update(kwargs)
            return fake_response

        app.state.orchestrator.run = spy_run

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/ask",
                json={
                    "question": "Tell me more",
                    "session_id": "sess-2",
                },
            )
        assert r.status_code == 200

        # The orchestrator must have received the prior history
        assert "history" in captured_kwargs
        assert captured_kwargs["history"] is not None
        assert len(captured_kwargs["history"]) == 2
        assert captured_kwargs["history"][0]["content"] == "What is FastAPI?"
        assert captured_kwargs["history"][1]["content"] == "FastAPI is a web framework."
