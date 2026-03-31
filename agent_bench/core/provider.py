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


class SelfHostedProvider(LLMProvider):
    """Provider targeting any OpenAI-compatible endpoint (vLLM, TGI, Ollama).

    Reads settings from config (provider.selfhosted.*) with env var fallback:
        MODAL_VLLM_URL   -> base_url
        SELFHOSTED_MODEL -> model_name
        MODAL_AUTH_TOKEN  -> api_key

    Tool-calling support is detected lazily on the first complete() call
    with tools. If the endpoint returns a 400 or the model ignores tools,
    subsequent calls fall back to prompt-based tool selection.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        import os

        import httpx as _httpx

        self.config = config or load_config()
        sh = self.config.provider.selfhosted
        self.base_url = (
            sh.base_url
            or os.environ.get("MODAL_VLLM_URL", "http://localhost:8001/v1")
        )
        self.model = (
            sh.model_name
            if sh.model_name != "mistralai/Mistral-7B-Instruct-v0.3"
            else os.environ.get("SELFHOSTED_MODEL", sh.model_name)
        )
        api_key = sh.api_key or os.environ.get("MODAL_AUTH_TOKEN", "")
        self._supports_tool_calling: bool | None = None  # detected lazily

        model_pricing = self.config.provider.models.get(self.model)
        self._input_cost = model_pricing.input_cost_per_mtok if model_pricing else 0.0
        self._output_cost = model_pricing.output_cost_per_mtok if model_pricing else 0.0

        self.client = _httpx.AsyncClient(
            base_url=self.base_url,
            timeout=sh.timeout_seconds,
            follow_redirects=True,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    async def _detect_tool_calling(self) -> bool | None:
        """Probe the endpoint for OpenAI-format tool-calling support.

        Returns:
            True  — model responded with tool_calls (definitive: cache it)
            False — endpoint returned 400 (definitive: cache it)
            None  — transient failure (timeout, 5xx, connection error); do NOT cache
        """
        test_tool = {
            "type": "function",
            "function": {
                "name": "test_probe",
                "description": "Probe for tool support",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            },
        }
        try:
            resp = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": "Call the test_probe tool with x='hello'"}
                    ],
                    "tools": [test_tool],
                    "tool_choice": "auto",
                    "max_tokens": 50,
                },
            )
            if resp.status_code == 400:
                log.info("selfhosted_tool_detect", result="unsupported (400)")
                return False
            if resp.status_code >= 500:
                log.warning("selfhosted_tool_detect", result="transient (5xx)")
                return None
            resp.raise_for_status()
            data = resp.json()
            has_tools = bool(
                data["choices"][0]["message"].get("tool_calls")
            )
            log.info("selfhosted_tool_detect", result="supported" if has_tools else "unsupported")
            return has_tools
        except Exception:
            log.warning("selfhosted_tool_detect", result="transient (error)")
            return None

    @staticmethod
    def _sanitize_messages(messages: list[dict]) -> list[dict]:
        """Convert tool-role messages and merge consecutive same-role messages.

        Many models (e.g. Mistral) require strictly alternating user/assistant
        messages. Tool results are converted to user messages and consecutive
        same-role messages are merged.
        """
        sanitized = []
        for m in messages:
            if m["role"] == "tool":
                role = "user"
                content = f"[Tool result]: {m['content']}"
            elif m["role"] == "assistant" and "tool_calls" in m:
                role = "assistant"
                content = m.get("content") or ""
            else:
                role = m["role"]
                content = m.get("content") or ""

            # Merge consecutive same-role messages
            if sanitized and sanitized[-1]["role"] == role and role != "system":
                sanitized[-1]["content"] += "\n\n" + content
            else:
                sanitized.append({"role": role, "content": content})

        # Merge consecutive same-role messages that resulted from dropping empty ones
        merged = []
        for m in sanitized:
            if not m["content"].strip() and m["role"] != "system":
                continue  # drop empty messages
            if merged and merged[-1]["role"] == m["role"] and m["role"] != "system":
                merged[-1]["content"] += "\n\n" + m["content"]
            else:
                merged.append(m)
        return merged

    @staticmethod
    def _tools_as_prompt(tools: list[ToolDefinition]) -> str:
        """Format tools as system prompt text for prompt-based fallback."""
        lines = ["You have access to the following tools:", ""]
        for t in tools:
            lines.append(f"- {t.name}: {t.description}")
            lines.append(f"  Parameters: {json.dumps(t.parameters)}")
        lines.extend([
            "",
            "To use a tool, respond with ONLY this JSON (no other text):",
            '{"tool_calls": [{"name": "tool_name", "arguments": {"param": "value"}}]}',
            "",
            "If you don't need a tool, respond normally with text.",
        ])
        return "\n".join(lines)

    @staticmethod
    def _parse_tool_calls_from_text(text: str) -> list[ToolCall]:
        """Parse tool calls from model text output (prompt-based fallback)."""
        import uuid

        try:
            data = json.loads(text.strip())
            if isinstance(data, dict) and "tool_calls" in data:
                calls = []
                for tc in data["tool_calls"]:
                    raw_args = tc.get("arguments", {})
                    if not isinstance(raw_args, dict):
                        raw_args = {}
                    calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=tc["name"],
                            arguments=raw_args,
                        )
                    )
                return calls
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResponse:
        import httpx as _httpx

        # Lazy tool-calling detection on first call with tools
        if tools and self._supports_tool_calling is None:
            result = await self._detect_tool_calling()
            if result is not None:
                self._supports_tool_calling = result
            # If None (transient), leave as None so next call retries

        formatted_messages = format_messages_openai(messages)

        # Use native tools only when detection confirmed support.
        # When detection is None (transient failure), fall back to prompt-based
        # rather than risk a 400 with native tools on an unsupported endpoint.
        use_native_tools = tools and self._supports_tool_calling is True
        if tools and not use_native_tools:
            tool_prompt = self._tools_as_prompt(tools)
            # Merge tool instructions into existing system message (some models
            # like Mistral reject multiple system messages in their chat template)
            if formatted_messages and formatted_messages[0]["role"] == "system":
                formatted_messages[0]["content"] = (
                    tool_prompt + "\n\n" + formatted_messages[0]["content"]
                )
            else:
                formatted_messages = [
                    {"role": "system", "content": tool_prompt},
                    *formatted_messages,
                ]
        # Always sanitize for self-hosted: messages may contain tool/tool_calls
        # from earlier iterations even when current call has tools=None
        formatted_messages = self._sanitize_messages(formatted_messages)

        payload: dict = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if use_native_tools and tools:
            payload["tools"] = self.format_tools(tools)
            payload["tool_choice"] = "auto"

        retry_cfg = self.config.retry
        start = time.perf_counter()

        for attempt in range(retry_cfg.max_retries + 1):
            try:
                resp = await self.client.post("/chat/completions", json=payload)
                if resp.status_code == 429:
                    if attempt == retry_cfg.max_retries:
                        raise ProviderRateLimitError(
                            f"Rate limited after {retry_cfg.max_retries} retries"
                        )
                    wait = min(
                        retry_cfg.base_delay * (2 ** attempt), retry_cfg.max_delay
                    )
                    log.warning(
                        "selfhosted_retry",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code >= 400:
                    log.error("selfhosted_error", status=resp.status_code, body=resp.text[:500])
                resp.raise_for_status()
                break
            except _httpx.TimeoutException as e:
                raise ProviderTimeoutError(f"Self-hosted timed out: {e}") from e

        latency_ms = (time.perf_counter() - start) * 1000
        data = resp.json()

        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        tool_calls: list[ToolCall] = []

        if choice["message"].get("tool_calls"):
            # Native tool calling response
            for tc in choice["message"]["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=args,
                    )
                )
        elif tools and not self._supports_tool_calling and content:
            # Prompt-based fallback: parse tool calls from text
            tool_calls = self._parse_tool_calls_from_text(content)
            if tool_calls:
                content = ""  # tool call replaces text content

        usage_data = data.get("usage", {})
        input_tokens = usage_data.get("prompt_tokens", 0)
        output_tokens = usage_data.get("completion_tokens", 0)
        cost = (
            input_tokens * self._input_cost + output_tokens * self._output_cost
        ) / 1_000_000

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
            ),
            provider="selfhosted",
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
        import httpx as _httpx

        # Same tool-calling detection/fallback as complete()
        if tools and self._supports_tool_calling is None:
            result = await self._detect_tool_calling()
            if result is not None:
                self._supports_tool_calling = result

        formatted_messages = format_messages_openai(messages)
        use_native_tools = tools and self._supports_tool_calling is True
        if tools and not use_native_tools:
            tool_prompt = self._tools_as_prompt(tools)
            if formatted_messages and formatted_messages[0]["role"] == "system":
                formatted_messages[0]["content"] = (
                    tool_prompt + "\n\n" + formatted_messages[0]["content"]
                )
            else:
                formatted_messages = [
                    {"role": "system", "content": tool_prompt},
                    *formatted_messages,
                ]
        formatted_messages = self._sanitize_messages(formatted_messages)

        payload: dict = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if use_native_tools and tools:
            payload["tools"] = self.format_tools(tools)
            payload["tool_choice"] = "auto"

        retry_cfg = self.config.retry
        for attempt in range(retry_cfg.max_retries + 1):
            try:
                async with self.client.stream(
                    "POST", "/chat/completions", json=payload
                ) as resp:
                    if resp.status_code == 429:
                        if attempt == retry_cfg.max_retries:
                            raise ProviderRateLimitError(
                                f"Rate limited after {retry_cfg.max_retries} retries"
                            )
                        wait = min(
                            retry_cfg.base_delay * (2 ** attempt),
                            retry_cfg.max_delay,
                        )
                        log.warning(
                            "selfhosted_stream_retry",
                            attempt=attempt + 1,
                            wait_seconds=wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()

                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[len("data: "):]
                        if data_str == "[DONE]":
                            return
                        try:
                            chunk_data = json.loads(data_str)
                            delta = chunk_data["choices"][0].get("delta", {})
                            if delta.get("content"):
                                yield delta["content"]
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                return  # success — exit retry loop
            except _httpx.TimeoutException as e:
                raise ProviderTimeoutError(f"Self-hosted timed out: {e}") from e

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return format_tools_openai(tools)


def create_provider(config: AppConfig | None = None) -> LLMProvider:
    """Factory: create provider based on config."""
    if config is None:
        config = load_config()
    name = config.provider.default
    if name == "openai":
        return OpenAIProvider(config)
    elif name == "anthropic":
        return AnthropicProvider(config)
    elif name == "selfhosted":
        return SelfHostedProvider(config)
    elif name == "mock":
        return MockProvider()
    else:
        raise ValueError(f"Unknown provider: {name}")
