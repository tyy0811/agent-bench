"""Tests for core types, config, and provider abstraction."""

import pytest

from agent_bench.core.config import AppConfig, ProviderConfig, load_config, load_task_config
from agent_bench.core.provider import (
    AnthropicProvider,
    MockProvider,
    create_provider,
    format_messages_openai,
    format_tools_openai,
)
from agent_bench.core.types import (
    CompletionResponse,
    Message,
    Role,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

# --- Core types ---


class TestCoreTypes:
    def test_message_creation(self):
        msg = Message(role=Role.USER, content="hello")
        assert msg.role == Role.USER
        assert msg.content == "hello"
        assert msg.tool_call_id is None
        assert msg.tool_calls is None

    def test_tool_call_creation(self):
        tc = ToolCall(id="call_123", name="search", arguments={"query": "test"})
        assert tc.id == "call_123"
        assert tc.name == "search"
        assert tc.arguments == {"query": "test"}

    def test_token_usage_creation(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50, estimated_cost_usd=0.001)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.estimated_cost_usd == pytest.approx(0.001)

    def test_completion_response_defaults(self):
        resp = CompletionResponse(
            content="answer",
            usage=TokenUsage(input_tokens=10, output_tokens=5, estimated_cost_usd=0.0),
            provider="mock",
            model="mock-1",
            latency_ms=50.0,
        )
        assert resp.tool_calls == []
        assert resp.content == "answer"

    def test_tool_definition_schema(self):
        td = ToolDefinition(
            name="calculator",
            description="Evaluate math",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )
        assert td.name == "calculator"
        assert "expression" in td.parameters["properties"]


# --- Config ---


class TestConfig:
    def test_load_default_config(self):
        config = load_config()
        assert config.provider.default == "openai"
        assert config.agent.max_iterations == 3
        assert config.agent.temperature == 0.0
        assert config.rag.chunking.strategy == "recursive"
        assert config.rag.chunking.chunk_size == 512
        assert config.rag.retrieval.rrf_k == 60
        assert config.rag.retrieval.top_k == 5

    def test_model_pricing_available(self):
        config = load_config()
        models = config.provider.models
        assert "gpt-4o-mini" in models
        assert models["gpt-4o-mini"].input_cost_per_mtok == 0.15
        assert models["gpt-4o-mini"].output_cost_per_mtok == 0.60

    def test_cost_calculation(self):
        config = load_config()
        model_config = config.provider.models["gpt-4o-mini"]
        input_tokens = 1000
        output_tokens = 500
        expected_cost = (1000 * 0.15 + 500 * 0.60) / 1_000_000
        cost = (
            input_tokens * model_config.input_cost_per_mtok
            + output_tokens * model_config.output_cost_per_mtok
        ) / 1_000_000
        assert cost == pytest.approx(expected_cost)

    def test_load_task_config(self):
        task = load_task_config("tech_docs")
        assert task.name == "tech_docs"
        assert "search_documents" in task.system_prompt
        assert "[source:" in task.system_prompt


# --- MockProvider ---


class TestMockProvider:
    @pytest.mark.asyncio
    async def test_returns_tool_calls_on_first_call(self, mock_provider):
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Search for FastAPI path params"),
        ]
        tools = [
            ToolDefinition(
                name="search_documents",
                description="Search docs",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]
        response = await mock_provider.complete(messages, tools=tools)
        assert len(response.tool_calls) > 0
        assert response.tool_calls[0].name == "search_documents"
        assert response.provider == "mock"
        assert response.usage.input_tokens > 0

    @pytest.mark.asyncio
    async def test_returns_final_answer_when_tool_results_present(self, mock_provider):
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Search for FastAPI path params"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1", name="search_documents", arguments={"query": "path params"}
                    )
                ],
            ),
            Message(role=Role.TOOL, content="Path params use curly braces.", tool_call_id="call_1"),
        ]
        response = await mock_provider.complete(messages)
        assert response.tool_calls == []
        assert len(response.content) > 0
        assert response.usage.input_tokens > 0

    @pytest.mark.asyncio
    async def test_returns_answer_without_tools(self, mock_provider):
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello"),
        ]
        response = await mock_provider.complete(messages, tools=None)
        assert response.tool_calls == []
        assert len(response.content) > 0

    def test_format_tools_returns_list(self, mock_provider):
        tools = [
            ToolDefinition(
                name="calc",
                description="Calculate",
                parameters={"type": "object", "properties": {}},
            )
        ]
        formatted = mock_provider.format_tools(tools)
        assert isinstance(formatted, list)
        assert len(formatted) == 1


# --- OpenAI format functions (tested as pure functions, no API key needed) ---


class TestOpenAIFormat:
    def test_format_tools_produces_openai_schema(self):
        tools = [
            ToolDefinition(
                name="search_documents",
                description="Search the documentation corpus",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "description": "Number of results"},
                    },
                    "required": ["query"],
                },
            )
        ]
        formatted = format_tools_openai(tools)
        assert len(formatted) == 1
        assert formatted[0]["type"] == "function"
        func = formatted[0]["function"]
        assert func["name"] == "search_documents"
        assert func["description"] == "Search the documentation corpus"
        assert func["parameters"]["required"] == ["query"]

    def test_format_messages_maps_roles(self):
        messages = [
            Message(role=Role.SYSTEM, content="system prompt"),
            Message(role=Role.USER, content="user question"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[ToolCall(id="call_1", name="search", arguments={"q": "test"})],
            ),
            Message(role=Role.TOOL, content="tool result", tool_call_id="call_1"),
        ]
        formatted = format_messages_openai(messages)
        assert formatted[0]["role"] == "system"
        assert formatted[1]["role"] == "user"
        assert formatted[2]["role"] == "assistant"
        assert formatted[2]["tool_calls"][0]["id"] == "call_1"
        assert formatted[2]["tool_calls"][0]["function"]["name"] == "search"
        assert formatted[3]["role"] == "tool"
        assert formatted[3]["tool_call_id"] == "call_1"


# --- Anthropic stub ---


class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_complete_raises_not_implemented(self):
        provider = AnthropicProvider()
        with pytest.raises(NotImplementedError, match="planned for V2"):
            await provider.complete([Message(role=Role.USER, content="test")])

    def test_format_tools_raises_not_implemented(self):
        provider = AnthropicProvider()
        with pytest.raises(NotImplementedError, match="planned for V2"):
            provider.format_tools([])


# --- Provider factory ---


class TestProviderFactory:
    def test_create_mock_provider(self):
        config = AppConfig(provider=ProviderConfig(default="mock"))
        provider = create_provider(config)
        assert isinstance(provider, MockProvider)

    def test_create_unknown_provider_raises(self):
        config = AppConfig(provider=ProviderConfig(default="unknown"))
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider(config)
