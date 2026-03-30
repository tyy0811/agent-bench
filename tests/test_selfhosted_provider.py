"""Tests for the SelfHostedProvider (OpenAI-compatible endpoint)."""

import json

import httpx
import pytest
import respx

from agent_bench.core.config import (
    AppConfig,
    ProviderConfig,
    RetryConfig,
    SelfHostedConfig,
)
from agent_bench.core.provider import (
    ProviderRateLimitError,
    ProviderTimeoutError,
    SelfHostedProvider,
    create_provider,
)
from agent_bench.core.types import Message, Role, ToolDefinition

# --- Helpers ---

FAKE_URL = "http://fake-vllm:8000/v1"

SEARCH_TOOL = ToolDefinition(
    name="search_documents",
    description="Search docs",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
)


def _ok_response(content="ok", tool_calls=None, prompt_tokens=10, completion_tokens=5):
    """Build a minimal OpenAI-format chat completion response."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _probe_response_with_tool_calls():
    """Response to the tool-calling detection probe — model uses tools."""
    return _ok_response(
        tool_calls=[
            {
                "id": "call_probe",
                "type": "function",
                "function": {
                    "name": "test_probe",
                    "arguments": json.dumps({"x": "hello"}),
                },
            }
        ],
    )


def _probe_response_without_tool_calls():
    """Response to the tool-calling detection probe — model ignores tools."""
    return _ok_response(content="I cannot use tools.")


# --- Factory ---


class TestSelfHostedFactory:
    def test_factory_creates_selfhosted_provider(self, monkeypatch):
        """Factory returns SelfHostedProvider for 'selfhosted' config."""
        monkeypatch.setenv("MODAL_VLLM_URL", FAKE_URL)
        config = AppConfig(provider=ProviderConfig(default="selfhosted"))
        provider = create_provider(config)
        assert isinstance(provider, SelfHostedProvider)

    def test_factory_raises_for_unknown_provider(self):
        config = AppConfig(provider=ProviderConfig(default="nonexistent"))
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider(config)


# --- Config-based settings ---


class TestSelfHostedConfig:
    def test_reads_base_url_from_config(self, monkeypatch):
        """Config selfhosted.base_url takes precedence over env var."""
        monkeypatch.setenv("MODAL_VLLM_URL", "http://env-url:8000/v1")
        config = AppConfig(
            provider=ProviderConfig(
                default="selfhosted",
                selfhosted=SelfHostedConfig(base_url="http://config-url:8000/v1"),
            )
        )
        provider = SelfHostedProvider(config)
        assert provider.base_url == "http://config-url:8000/v1"

    def test_falls_back_to_env_when_config_empty(self, monkeypatch):
        """Empty config falls back to MODAL_VLLM_URL env var."""
        monkeypatch.setenv("MODAL_VLLM_URL", "http://env-url:8000/v1")
        config = AppConfig(provider=ProviderConfig(default="selfhosted"))
        provider = SelfHostedProvider(config)
        assert provider.base_url == "http://env-url:8000/v1"

    def test_reads_api_key_from_config(self, monkeypatch):
        monkeypatch.delenv("MODAL_AUTH_TOKEN", raising=False)
        config = AppConfig(
            provider=ProviderConfig(
                default="selfhosted",
                selfhosted=SelfHostedConfig(
                    base_url=FAKE_URL, api_key="config-key-123"
                ),
            )
        )
        provider = SelfHostedProvider(config)
        assert provider.client.headers.get("authorization") == "Bearer config-key-123"

    def test_timeout_from_config(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", FAKE_URL)
        config = AppConfig(
            provider=ProviderConfig(
                default="selfhosted",
                selfhosted=SelfHostedConfig(timeout_seconds=42.0),
            )
        )
        provider = SelfHostedProvider(config)
        assert provider.client.timeout.read == 42.0

    def test_config_yaml_selfhosted_block_not_dropped(self):
        """Pydantic accepts provider.selfhosted fields (regression for issue #3)."""
        raw = {
            "provider": {
                "default": "selfhosted",
                "selfhosted": {
                    "base_url": "http://yaml-url:8000/v1",
                    "model_name": "meta-llama/Llama-3-8B",
                    "api_key": "yaml-key",
                    "timeout_seconds": 60.0,
                },
            }
        }
        config = AppConfig.model_validate(raw)
        assert config.provider.selfhosted.base_url == "http://yaml-url:8000/v1"
        assert config.provider.selfhosted.model_name == "meta-llama/Llama-3-8B"
        assert config.provider.selfhosted.api_key == "yaml-key"
        assert config.provider.selfhosted.timeout_seconds == 60.0


# --- complete() ---


class TestSelfHostedComplete:
    @pytest.fixture
    def provider(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", FAKE_URL)
        config = AppConfig(provider=ProviderConfig(default="selfhosted"))
        return SelfHostedProvider(config)

    @pytest.mark.asyncio
    async def test_complete_parses_response(self, provider):
        """SelfHostedProvider.complete() parses OpenAI-format response."""
        mock_response = _ok_response(
            content="Path params use curly braces. [source: fastapi.md]",
            prompt_tokens=80,
            completion_tokens=20,
        )

        with respx.mock:
            respx.post(f"{FAKE_URL}/chat/completions").mock(
                return_value=httpx.Response(200, json=mock_response)
            )
            response = await provider.complete(
                [Message(role=Role.USER, content="How do path params work?")]
            )

        assert response.content == "Path params use curly braces. [source: fastapi.md]"
        assert response.tool_calls == []
        assert response.provider == "selfhosted"
        assert response.model == "mistralai/Mistral-7B-Instruct-v0.3"
        assert response.usage.input_tokens == 80
        assert response.usage.output_tokens == 20
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_complete_parses_tool_calls(self, provider):
        """SelfHostedProvider.complete() parses native tool_calls."""
        # Pre-set tool support to skip detection probe
        provider._supports_tool_calling = True

        tool_response = _ok_response(
            tool_calls=[
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "search_documents",
                        "arguments": json.dumps({"query": "path params"}),
                    },
                }
            ],
            prompt_tokens=60,
            completion_tokens=15,
        )

        with respx.mock:
            respx.post(f"{FAKE_URL}/chat/completions").mock(
                return_value=httpx.Response(200, json=tool_response)
            )
            response = await provider.complete(
                [Message(role=Role.USER, content="search for path params")],
                tools=[SEARCH_TOOL],
            )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_abc"
        assert response.tool_calls[0].name == "search_documents"
        assert response.tool_calls[0].arguments == {"query": "path params"}

    @pytest.mark.asyncio
    async def test_complete_handles_malformed_tool_args(self, provider):
        """Malformed JSON in tool arguments falls back to empty dict."""
        provider._supports_tool_calling = True

        mock_response = _ok_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {
                        "name": "search_documents",
                        "arguments": "not valid json{{{",
                    },
                }
            ],
        )

        with respx.mock:
            respx.post(f"{FAKE_URL}/chat/completions").mock(
                return_value=httpx.Response(200, json=mock_response)
            )
            response = await provider.complete(
                [Message(role=Role.USER, content="test")]
            )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].arguments == {}


# --- Tool-calling detection ---


class TestSelfHostedToolDetection:
    @pytest.fixture
    def provider(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", FAKE_URL)
        config = AppConfig(provider=ProviderConfig(default="selfhosted"))
        return SelfHostedProvider(config)

    @pytest.mark.asyncio
    async def test_detect_tool_calling_supported(self, provider):
        """Detection probe returns True when model responds with tool_calls."""
        with respx.mock:
            respx.post(f"{FAKE_URL}/chat/completions").mock(
                return_value=httpx.Response(
                    200, json=_probe_response_with_tool_calls()
                )
            )
            result = await provider._detect_tool_calling()
        assert result is True

    @pytest.mark.asyncio
    async def test_detect_tool_calling_unsupported_400(self, provider):
        """Detection probe returns False on 400 (endpoint rejects tools)."""
        with respx.mock:
            respx.post(f"{FAKE_URL}/chat/completions").mock(
                return_value=httpx.Response(
                    400, json={"error": "tools not supported"}
                )
            )
            result = await provider._detect_tool_calling()
        assert result is False

    @pytest.mark.asyncio
    async def test_detect_tool_calling_unsupported_no_tool_calls(self, provider):
        """Detection probe returns False when model ignores tools."""
        with respx.mock:
            respx.post(f"{FAKE_URL}/chat/completions").mock(
                return_value=httpx.Response(
                    200, json=_probe_response_without_tool_calls()
                )
            )
            result = await provider._detect_tool_calling()
        assert result is False

    @pytest.mark.asyncio
    async def test_detection_runs_once_then_cached(self, provider):
        """Detection probe fires on first call with tools, cached thereafter."""
        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            body = json.loads(request.content)
            # Detection probe has test_probe tool
            if any(
                t.get("function", {}).get("name") == "test_probe"
                for t in body.get("tools", [])
            ):
                return httpx.Response(
                    200, json=_probe_response_with_tool_calls()
                )
            # Real request
            return httpx.Response(200, json=_ok_response(
                tool_calls=[{
                    "id": "call_real",
                    "type": "function",
                    "function": {
                        "name": "search_documents",
                        "arguments": json.dumps({"query": "test"}),
                    },
                }],
            ))

        with respx.mock:
            respx.post(f"{FAKE_URL}/chat/completions").mock(
                side_effect=side_effect
            )
            # First call: probe + real = 2 requests
            await provider.complete(
                [Message(role=Role.USER, content="test")],
                tools=[SEARCH_TOOL],
            )
            # Second call: no probe = 1 request
            await provider.complete(
                [Message(role=Role.USER, content="test2")],
                tools=[SEARCH_TOOL],
            )

        assert call_count == 3  # 1 probe + 2 real
        assert provider._supports_tool_calling is True


# --- Prompt-based fallback ---


class TestSelfHostedPromptFallback:
    @pytest.fixture
    def provider(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", FAKE_URL)
        config = AppConfig(provider=ProviderConfig(default="selfhosted"))
        p = SelfHostedProvider(config)
        p._supports_tool_calling = False  # Force fallback mode
        return p

    @pytest.mark.asyncio
    async def test_fallback_parses_tool_call_from_text(self, provider):
        """When tool calling is unsupported, parse tool calls from model text."""
        tool_json = json.dumps(
            {"tool_calls": [{"name": "search_documents", "arguments": {"query": "path params"}}]}
        )
        mock_response = _ok_response(content=tool_json)

        with respx.mock:
            route = respx.post(f"{FAKE_URL}/chat/completions").mock(
                return_value=httpx.Response(200, json=mock_response)
            )
            response = await provider.complete(
                [Message(role=Role.USER, content="search for path params")],
                tools=[SEARCH_TOOL],
            )
            # Verify tools NOT in payload (prompt-based, not native)
            sent_body = json.loads(route.calls[0].request.content)
            assert "tools" not in sent_body

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search_documents"
        assert response.tool_calls[0].arguments == {"query": "path params"}
        assert response.content == ""  # tool call replaces content

    @pytest.mark.asyncio
    async def test_fallback_injects_tool_prompt(self, provider):
        """When tool calling is unsupported, tool descriptions injected as system prompt."""
        mock_response = _ok_response(content="Just a text answer.")

        with respx.mock:
            route = respx.post(f"{FAKE_URL}/chat/completions").mock(
                return_value=httpx.Response(200, json=mock_response)
            )
            await provider.complete(
                [Message(role=Role.USER, content="hello")],
                tools=[SEARCH_TOOL],
            )
            sent_body = json.loads(route.calls[0].request.content)

        # System message should contain tool descriptions
        system_msg = sent_body["messages"][0]
        assert system_msg["role"] == "system"
        assert "search_documents" in system_msg["content"]
        assert "tool_calls" in system_msg["content"]

    @pytest.mark.asyncio
    async def test_fallback_returns_text_when_no_tool_json(self, provider):
        """When model responds with plain text (not JSON), return as content."""
        mock_response = _ok_response(content="I don't know how to use tools.")

        with respx.mock:
            respx.post(f"{FAKE_URL}/chat/completions").mock(
                return_value=httpx.Response(200, json=mock_response)
            )
            response = await provider.complete(
                [Message(role=Role.USER, content="test")],
                tools=[SEARCH_TOOL],
            )

        assert response.tool_calls == []
        assert response.content == "I don't know how to use tools."


# --- Retry and timeout ---


class TestSelfHostedRetryAndTimeout:
    @pytest.fixture
    def provider(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", FAKE_URL)
        config = AppConfig(
            provider=ProviderConfig(default="selfhosted"),
            retry=RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.05),
        )
        return SelfHostedProvider(config)

    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self, provider):
        """Provider retries on 429 and succeeds on next attempt."""
        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json=_ok_response())

        with respx.mock:
            respx.post(f"{FAKE_URL}/chat/completions").mock(
                side_effect=side_effect
            )
            response = await provider.complete(
                [Message(role=Role.USER, content="test")]
            )

        assert response.content == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_raises_rate_limit_after_exhausting_retries(self, provider):
        """Provider raises ProviderRateLimitError after all retries exhausted."""
        with respx.mock:
            respx.post(f"{FAKE_URL}/chat/completions").mock(
                return_value=httpx.Response(429, json={"error": "rate limited"})
            )
            with pytest.raises(ProviderRateLimitError, match="Rate limited"):
                await provider.complete(
                    [Message(role=Role.USER, content="test")]
                )

    @pytest.mark.asyncio
    async def test_raises_timeout_error(self, provider):
        """Provider raises ProviderTimeoutError on httpx timeout."""
        with respx.mock:
            respx.post(f"{FAKE_URL}/chat/completions").mock(
                side_effect=httpx.ReadTimeout("timed out")
            )
            with pytest.raises(ProviderTimeoutError, match="timed out"):
                await provider.complete(
                    [Message(role=Role.USER, content="test")]
                )


# --- Env var fallback ---


class TestSelfHostedEnvVars:
    def test_reads_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", "http://my-modal-url:8000/v1")
        config = AppConfig(provider=ProviderConfig(default="selfhosted"))
        provider = SelfHostedProvider(config)
        assert provider.base_url == "http://my-modal-url:8000/v1"

    def test_reads_auth_token_from_env(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", FAKE_URL)
        monkeypatch.setenv("MODAL_AUTH_TOKEN", "secret-token-123")
        config = AppConfig(provider=ProviderConfig(default="selfhosted"))
        provider = SelfHostedProvider(config)
        assert provider.client.headers.get("authorization") == "Bearer secret-token-123"

    def test_no_auth_header_when_no_token(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", FAKE_URL)
        monkeypatch.delenv("MODAL_AUTH_TOKEN", raising=False)
        config = AppConfig(provider=ProviderConfig(default="selfhosted"))
        provider = SelfHostedProvider(config)
        assert "authorization" not in {
            k.lower() for k in provider.client.headers.keys()
        }


# --- Streaming ---


class TestSelfHostedStream:
    @pytest.fixture
    def provider(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", FAKE_URL)
        config = AppConfig(provider=ProviderConfig(default="selfhosted"))
        return SelfHostedProvider(config)

    @pytest.mark.asyncio
    async def test_stream_yields_content_chunks(self, provider):
        """stream_complete() yields text chunks from SSE stream."""
        sse_body = (
            'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
            "data: [DONE]\n\n"
        )

        with respx.mock:
            respx.post(f"{FAKE_URL}/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    stream=httpx.ByteStream(sse_body.encode()),
                    headers={"content-type": "text/event-stream"},
                )
            )
            chunks = []
            async for chunk in provider.stream_complete(
                [Message(role=Role.USER, content="Hi")]
            ):
                chunks.append(chunk)

        assert chunks == ["Hello ", "world"]


# --- format_tools ---


class TestSelfHostedFormatTools:
    def test_format_tools_uses_openai_schema(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", FAKE_URL)
        config = AppConfig(provider=ProviderConfig(default="selfhosted"))
        provider = SelfHostedProvider(config)
        tools = [
            ToolDefinition(
                name="search_documents",
                description="Search docs",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ]
        formatted = provider.format_tools(tools)
        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "search_documents"
        assert formatted[0]["function"]["parameters"]["required"] == ["query"]
