"""Tests for core types, config, and provider abstraction."""

from unittest.mock import patch

import pytest

from agent_bench.core.config import (
    AppConfig,
    ProviderConfig,
    RetryConfig,
    load_config,
    load_task_config,
)
from agent_bench.core.provider import (
    AnthropicProvider,
    MockProvider,
    ProviderRateLimitError,
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


# --- OpenAI provider (mocked HTTP) ---


class TestOpenAIProvider:
    def test_factory_creates_openai_provider(self, monkeypatch):
        """Factory returns OpenAIProvider for 'openai' config."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")
        from agent_bench.core.provider import OpenAIProvider

        config = AppConfig(provider=ProviderConfig(default="openai"))
        provider = create_provider(config)
        assert isinstance(provider, OpenAIProvider)

    def test_format_tools_via_instance(self, monkeypatch):
        """OpenAIProvider.format_tools delegates to format_tools_openai correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")
        from agent_bench.core.provider import OpenAIProvider

        config = AppConfig(provider=ProviderConfig(default="openai"))
        provider = OpenAIProvider(config)
        tools = [
            ToolDefinition(
                name="search_documents",
                description="Search docs",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]
        formatted = provider.format_tools(tools)
        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "search_documents"

    @pytest.mark.asyncio
    async def test_complete_with_mocked_response(self, monkeypatch):
        """OpenAI complete() parses a mocked API response correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")

        import httpx
        import respx

        from agent_bench.core.provider import OpenAIProvider

        config = AppConfig(provider=ProviderConfig(default="openai"))
        provider = OpenAIProvider(config)

        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "FastAPI uses curly braces. [source: path_params.md]",
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
        }

        with respx.mock:
            respx.post("https://api.openai.com/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=mock_response)
            )
            response = await provider.complete(
                [Message(role=Role.USER, content="How do path params work?")]
            )

        assert response.content == "FastAPI uses curly braces. [source: path_params.md]"
        assert response.tool_calls == []
        assert response.provider == "openai"
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 30
        assert response.usage.estimated_cost_usd > 0
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_complete_parses_tool_calls(self, monkeypatch):
        """OpenAI complete() correctly parses tool_calls from response."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")
        import json

        import httpx
        import respx

        from agent_bench.core.provider import OpenAIProvider

        config = AppConfig(provider=ProviderConfig(default="openai"))
        provider = OpenAIProvider(config)

        mock_response = {
            "id": "chatcmpl-test2",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "search_documents",
                                    "arguments": json.dumps({"query": "path parameters"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
        }

        tools = [
            ToolDefinition(
                name="search_documents",
                description="Search docs",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]

        with respx.mock:
            respx.post("https://api.openai.com/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=mock_response)
            )
            response = await provider.complete(
                [Message(role=Role.USER, content="search for path params")],
                tools=tools,
            )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_abc123"
        assert response.tool_calls[0].name == "search_documents"
        assert response.tool_calls[0].arguments == {"query": "path parameters"}


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


# --- Retry logic ---


class TestProviderRetry:
    """Tests for OpenAI provider retry with exponential backoff."""

    MOCK_SUCCESS_RESPONSE = {
        "id": "chatcmpl-retry",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Success after retry.",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, monkeypatch):
        """Two failures then success — returns answer."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")

        import httpx
        import respx

        from agent_bench.core.provider import OpenAIProvider

        config = AppConfig(
            provider=ProviderConfig(default="openai"),
            retry=RetryConfig(max_retries=3, base_delay=0.01, max_delay=0.1),
        )
        provider = OpenAIProvider(config)

        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
            return httpx.Response(200, json=self.MOCK_SUCCESS_RESPONSE)

        with respx.mock:
            respx.post("https://api.openai.com/v1/chat/completions").mock(
                side_effect=side_effect
            )
            from agent_bench.core.types import Message, Role

            response = await provider.complete(
                [Message(role=Role.USER, content="test")]
            )

        assert response.content == "Success after retry."
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, monkeypatch):
        """All retries fail — raises ProviderRateLimitError."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")

        import httpx
        import respx

        from agent_bench.core.provider import OpenAIProvider

        config = AppConfig(
            provider=ProviderConfig(default="openai"),
            retry=RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.1),
        )
        provider = OpenAIProvider(config)

        with respx.mock:
            respx.post("https://api.openai.com/v1/chat/completions").mock(
                return_value=httpx.Response(429, json={"error": {"message": "Rate limit"}})
            )
            from agent_bench.core.types import Message, Role

            with pytest.raises(ProviderRateLimitError, match="Rate limited after"):
                await provider.complete(
                    [Message(role=Role.USER, content="test")]
                )

    @pytest.mark.asyncio
    async def test_no_retry_on_other_errors(self, monkeypatch):
        """Non-rate-limit errors fail immediately without retry."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")

        import httpx
        import respx

        from agent_bench.core.provider import OpenAIProvider

        config = AppConfig(
            provider=ProviderConfig(default="openai"),
            retry=RetryConfig(max_retries=3, base_delay=0.01, max_delay=0.1),
        )
        provider = OpenAIProvider(config)

        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            return httpx.Response(400, json={"error": {"message": "Bad request"}})

        with respx.mock:
            respx.post("https://api.openai.com/v1/chat/completions").mock(
                side_effect=side_effect
            )
            from agent_bench.core.types import Message, Role

            with pytest.raises(Exception):
                await provider.complete(
                    [Message(role=Role.USER, content="test")]
                )

        assert call_count == 1  # no retry

    @pytest.mark.asyncio
    async def test_retry_backoff_timing(self, monkeypatch):
        """Verify exponential backoff delays between retries."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")

        import httpx
        import respx

        from agent_bench.core.provider import OpenAIProvider

        config = AppConfig(
            provider=ProviderConfig(default="openai"),
            retry=RetryConfig(max_retries=3, base_delay=1.0, max_delay=8.0),
        )
        provider = OpenAIProvider(config)

        sleep_calls: list[float] = []

        async def mock_sleep(seconds):
            sleep_calls.append(seconds)

        with respx.mock, patch("asyncio.sleep", side_effect=mock_sleep):
            respx.post("https://api.openai.com/v1/chat/completions").mock(
                return_value=httpx.Response(429, json={"error": {"message": "Rate limit"}})
            )
            from agent_bench.core.types import Message, Role

            with pytest.raises(ProviderRateLimitError):
                await provider.complete(
                    [Message(role=Role.USER, content="test")]
                )

        # 3 retries: delays should be 1.0, 2.0, 4.0
        assert len(sleep_calls) == 3
        assert sleep_calls[0] == pytest.approx(1.0)
        assert sleep_calls[1] == pytest.approx(2.0)
        assert sleep_calls[2] == pytest.approx(4.0)


class TestStreamingRetry:
    """Tests for stream_complete() retry/timeout parity with complete()."""

    @pytest.mark.asyncio
    async def test_stream_retry_on_rate_limit(self, monkeypatch):
        """stream_complete retries on 429 then succeeds."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")

        import httpx
        import respx

        from agent_bench.core.provider import OpenAIProvider

        config = AppConfig(
            provider=ProviderConfig(default="openai"),
            retry=RetryConfig(max_retries=3, base_delay=0.01, max_delay=0.1),
        )
        provider = OpenAIProvider(config)

        call_count = 0

        # Streaming API: first 2 calls return 429, third returns SSE chunks
        def side_effect(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return httpx.Response(
                    429, json={"error": {"message": "Rate limit"}}
                )
            # Simulate streaming response with SSE format
            sse_body = (
                'data: {"id":"x","object":"chat.completion.chunk",'
                '"choices":[{"index":0,"delta":{"content":"hello"},'
                '"finish_reason":null}]}\n\n'
                'data: [DONE]\n\n'
            )
            return httpx.Response(
                200,
                content=sse_body.encode(),
                headers={"content-type": "text/event-stream"},
            )

        with respx.mock:
            respx.post("https://api.openai.com/v1/chat/completions").mock(
                side_effect=side_effect
            )
            from agent_bench.core.types import Message, Role

            chunks = []
            async for chunk in provider.stream_complete(
                [Message(role=Role.USER, content="test")]
            ):
                chunks.append(chunk)

        assert call_count == 3
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_stream_retry_exhausted(self, monkeypatch):
        """stream_complete raises ProviderRateLimitError after retries."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")

        import httpx
        import respx

        from agent_bench.core.provider import OpenAIProvider

        config = AppConfig(
            provider=ProviderConfig(default="openai"),
            retry=RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.1),
        )
        provider = OpenAIProvider(config)

        with respx.mock:
            respx.post("https://api.openai.com/v1/chat/completions").mock(
                return_value=httpx.Response(
                    429, json={"error": {"message": "Rate limit"}}
                )
            )
            from agent_bench.core.types import Message, Role

            with pytest.raises(ProviderRateLimitError, match="Rate limited"):
                async for _ in provider.stream_complete(
                    [Message(role=Role.USER, content="test")]
                ):
                    pass  # pragma: no cover

    @pytest.mark.asyncio
    async def test_stream_timeout_raises(self, monkeypatch):
        """stream_complete translates APITimeoutError to ProviderTimeoutError."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")

        from agent_bench.core.provider import OpenAIProvider, ProviderTimeoutError

        config = AppConfig(
            provider=ProviderConfig(default="openai"),
            retry=RetryConfig(max_retries=1, base_delay=0.01, max_delay=0.1),
        )
        provider = OpenAIProvider(config)

        from openai import APITimeoutError

        async def mock_create(**kwargs):
            raise APITimeoutError(request=None)

        provider.client.chat.completions.create = mock_create  # type: ignore[assignment]

        from agent_bench.core.types import Message, Role

        with pytest.raises(ProviderTimeoutError, match="timed out"):
            async for _ in provider.stream_complete(
                [Message(role=Role.USER, content="test")]
            ):
                pass  # pragma: no cover
