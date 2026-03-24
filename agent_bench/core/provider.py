"""LLM provider abstraction with OpenAI, Mock, and Anthropic (stub) implementations."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod

from agent_bench.core.config import AppConfig, load_config
from agent_bench.core.types import (
    CompletionResponse,
    Message,
    Role,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)


class ProviderTimeoutError(Exception):
    """Raised when the LLM provider times out."""


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
        from openai import APITimeoutError

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

        start = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(**kwargs)
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

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return format_tools_openai(tools)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider -- stub for V2."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResponse:
        raise NotImplementedError("Anthropic provider planned for V2")

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        raise NotImplementedError("Anthropic provider planned for V2")


def create_provider(config: AppConfig | None = None) -> LLMProvider:
    """Factory: create provider based on config."""
    if config is None:
        config = load_config()
    name = config.provider.default
    if name == "openai":
        return OpenAIProvider(config)
    elif name == "anthropic":
        return AnthropicProvider()
    elif name == "mock":
        return MockProvider()
    else:
        raise ValueError(f"Unknown provider: {name}")
