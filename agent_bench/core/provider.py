"""LLM provider abstraction with OpenAI, Mock, and Anthropic (stub) implementations."""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import structlog

from agent_bench.core.config import AppConfig, load_config
from agent_bench.core.types import (
    CompletionResponse,
    Message,
    Role,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

log = structlog.get_logger()


class ProviderTimeoutError(Exception):
    """Raised when the LLM provider times out."""


class ProviderRateLimitError(Exception):
    """Raised when the LLM provider returns a rate limit / quota error."""


# --- Pure formatting functions (used by providers and tests directly) ---


def format_tools_openai(tools: list[ToolDefinition]) -> list[dict]:
    """Format tool definitions into OpenAI function-calling schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def format_messages_openai(messages: list[Message]) -> list[dict]:
    """Format internal Message objects into OpenAI chat message dicts."""
    formatted = []
    for m in messages:
        msg: dict = {"role": m.role.value, "content": m.content}
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in m.tool_calls
            ]
        formatted.append(msg)
    return formatted


# --- Provider interface ---


class LLMProvider(ABC):
    """Async LLM provider interface."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResponse: ...

    @abstractmethod
    async def stream_complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Yield content chunks as they arrive from the provider."""
        ...
        yield ""  # pragma: no cover

    @abstractmethod
    def format_tools(self, tools: list[ToolDefinition]) -> list[dict]: ...


# --- Implementations ---


class MockProvider(LLMProvider):
    """Deterministic provider for testing.

    Behavior:
    - If tools are provided AND no Role.TOOL messages exist -> returns tool_calls
    - If Role.TOOL messages exist OR no tools -> returns final text answer
    """

    def __init__(self) -> None:
        self.call_count = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResponse:
        self.call_count += 1
        has_tool_results = any(m.role == Role.TOOL for m in messages)

        if tools and not has_tool_results:
            return CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"call_mock_{self.call_count}",
                        name=tools[0].name,
                        arguments={"query": "mock search query"},
                    )
                ],
                usage=TokenUsage(
                    input_tokens=150,
                    output_tokens=25,
                    estimated_cost_usd=0.0001,
                ),
                provider="mock",
                model="mock-1",
                latency_ms=1.0,
            )

        return CompletionResponse(
            content="Based on the documentation, path parameters in FastAPI are defined "
            "using curly braces in the path string. [source: fastapi_path_params.md]",
            tool_calls=[],
            usage=TokenUsage(
                input_tokens=200,
                output_tokens=50,
                estimated_cost_usd=0.0002,
            ),
            provider="mock",
            model="mock-1",
            latency_ms=2.0,
        )

    async def stream_complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Yield mock response in exactly 3 deterministic chunks."""
        chunks = [
            "Based on the documentation, path parameters in FastAPI ",
            "are defined using curly braces in the path string. ",
            "[source: fastapi_path_params.md]",
        ]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.01)

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return format_tools_openai(tools)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider using gpt-4o-mini."""

    def __init__(self, config: AppConfig | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install openai") from e

        import os

        self.config = config or load_config()
        api_key = os.environ.get("OPENAI_API_KEY", "")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        model_pricing = self.config.provider.models.get(self.model)
        self._input_cost = model_pricing.input_cost_per_mtok if model_pricing else 0.15
        self._output_cost = model_pricing.output_cost_per_mtok if model_pricing else 0.60

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResponse:
        from openai import APITimeoutError, RateLimitError

        formatted_messages = format_messages_openai(messages)
        kwargs: dict = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = self.format_tools(tools)
            kwargs["tool_choice"] = "auto"

        retry_cfg = self.config.retry
        start = time.perf_counter()

        for attempt in range(retry_cfg.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                break  # success
            except RateLimitError as e:
                if attempt == retry_cfg.max_retries:
                    log.error("provider_rate_limited",
                              attempts=attempt + 1, error=str(e))
                    raise ProviderRateLimitError(
                        f"Rate limited after {retry_cfg.max_retries} retries: {e}"
                    ) from e
                wait = min(
                    retry_cfg.base_delay * (2 ** attempt),
                    retry_cfg.max_delay,
                )
                log.warning("provider_retry",
                            attempt=attempt + 1, wait_seconds=wait, error=str(e))
                await asyncio.sleep(wait)
            except APITimeoutError as e:
                raise ProviderTimeoutError(f"OpenAI timed out: {e}") from e

        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls: list[ToolCall] = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = (input_tokens * self._input_cost + output_tokens * self._output_cost) / 1_000_000

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
            ),
            provider="openai",
            model=self.model,
            latency_ms=latency_ms,
        )

    async def stream_complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Yield content chunks from OpenAI streaming API.

        Same retry/timeout handling as complete() — wraps the raw openai
        call so 429s and timeouts are translated consistently.
        """
        from openai import APITimeoutError, RateLimitError

        formatted_messages = format_messages_openai(messages)
        kwargs: dict = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = self.format_tools(tools)
            kwargs["tool_choice"] = "auto"

        retry_cfg = self.config.retry
        for attempt in range(retry_cfg.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                break
            except RateLimitError as e:
                if attempt == retry_cfg.max_retries:
                    raise ProviderRateLimitError(
                        f"Rate limited after {retry_cfg.max_retries} retries: {e}"
                    ) from e
                wait = min(
                    retry_cfg.base_delay * (2 ** attempt),
                    retry_cfg.max_delay,
                )
                log.warning("provider_stream_retry",
                            attempt=attempt + 1, wait_seconds=wait)
                await asyncio.sleep(wait)
            except APITimeoutError as e:
                raise ProviderTimeoutError(f"OpenAI timed out: {e}") from e

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return format_tools_openai(tools)


def format_tools_anthropic(tools: list[ToolDefinition]) -> list[dict]:
    """Format tool definitions into Anthropic tool schema."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


def format_messages_anthropic(
    messages: list[Message],
) -> tuple[str, list[dict]]:
    """Extract system prompt and format messages for Anthropic API.

    Returns (system_prompt, messages) where system is separated out
    because Anthropic uses a system= parameter, not a message role.
    """
    system_prompt = ""
    formatted = []

    for m in messages:
        if m.role == Role.SYSTEM:
            system_prompt = m.content
            continue

        if m.role == Role.TOOL:
            # Anthropic: tool results are user messages with tool_result blocks
            formatted.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content,
                    }
                ],
            })
        elif m.role == Role.ASSISTANT and m.tool_calls:
            # Assistant message with tool_use content blocks
            content: list[dict] = []
            if m.content:
                content.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            formatted.append({"role": "assistant", "content": content})
        else:
            formatted.append({
                "role": m.role.value,
                "content": m.content,
            })

    return system_prompt, formatted


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, config: AppConfig | None = None) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required: pip install anthropic"
            ) from e

        import os

        self.config = config or load_config()
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = "claude-haiku-4-5-20251001"
        model_pricing = self.config.provider.models.get(self.model)
        self._input_cost = (
            model_pricing.input_cost_per_mtok if model_pricing else 0.80
        )
        self._output_cost = (
            model_pricing.output_cost_per_mtok if model_pricing else 4.0
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResponse:
        from anthropic import APITimeoutError, RateLimitError

        system_prompt, formatted_messages = format_messages_anthropic(messages)
        kwargs: dict = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = self.format_tools(tools)

        retry_cfg = self.config.retry
        start = time.perf_counter()

        for attempt in range(retry_cfg.max_retries + 1):
            try:
                response = await self.client.messages.create(**kwargs)
                break
            except RateLimitError as e:
                if attempt == retry_cfg.max_retries:
                    log.error(
                        "provider_rate_limited",
                        attempts=attempt + 1,
                        error=str(e),
                    )
                    raise ProviderRateLimitError(
                        f"Rate limited after {retry_cfg.max_retries} retries: {e}"
                    ) from e
                wait = min(
                    retry_cfg.base_delay * (2 ** attempt),
                    retry_cfg.max_delay,
                )
                log.warning(
                    "provider_retry",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                )
                await asyncio.sleep(wait)
            except APITimeoutError as e:
                raise ProviderTimeoutError(
                    f"Anthropic timed out: {e}"
                ) from e

        latency_ms = (time.perf_counter() - start) * 1000

        # Parse response content blocks
        content = ""
        tool_calls_out: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls_out.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = (
            input_tokens * self._input_cost
            + output_tokens * self._output_cost
        ) / 1_000_000

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls_out,
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
            ),
            provider="anthropic",
            model=self.model,
            latency_ms=latency_ms,
        )

    async def stream_complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Yield content chunks from Anthropic streaming API."""
        from anthropic import APITimeoutError, RateLimitError

        system_prompt, formatted_messages = format_messages_anthropic(messages)
        kwargs: dict = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = self.format_tools(tools)

        # Retry wraps the full stream lifecycle: .stream() returns a manager,
        # but the HTTP request fires in __aenter__. Both must be inside the
        # try/except to catch 429s and timeouts.
        retry_cfg = self.config.retry
        for attempt in range(retry_cfg.max_retries + 1):
            try:
                async with self.client.messages.stream(**kwargs) as s:
                    async for text in s.text_stream:
                        yield text
                return  # success — exit retry loop
            except RateLimitError as e:
                if attempt == retry_cfg.max_retries:
                    raise ProviderRateLimitError(
                        f"Rate limited after {retry_cfg.max_retries} retries: {e}"
                    ) from e
                wait = min(
                    retry_cfg.base_delay * (2 ** attempt),
                    retry_cfg.max_delay,
                )
                log.warning(
                    "provider_stream_retry",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                )
                await asyncio.sleep(wait)
            except APITimeoutError as e:
                raise ProviderTimeoutError(
                    f"Anthropic timed out: {e}"
                ) from e

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return format_tools_anthropic(tools)


def create_provider(config: AppConfig | None = None) -> LLMProvider:
    """Factory: create provider based on config."""
    if config is None:
        config = load_config()
    name = config.provider.default
    if name == "openai":
        return OpenAIProvider(config)
    elif name == "anthropic":
        return AnthropicProvider(config)
    elif name == "mock":
        return MockProvider()
    else:
        raise ValueError(f"Unknown provider: {name}")
