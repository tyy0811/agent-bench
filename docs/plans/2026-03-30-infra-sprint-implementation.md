# Infrastructure Sprint Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add self-hosted LLM serving (vLLM + Modal), Kubernetes Helm chart, and Terraform IaC to agent-bench.

**Architecture:** SelfHostedProvider targets any OpenAI-compatible endpoint (vLLM, TGI, Ollama) via httpx. GPU inference runs on Modal serverless A10G; K8s (Helm) handles the stateless API layer. Terraform provisions GCP/GKE for the API cluster. The provider detects tool-calling support via a startup smoke test.

**Tech Stack:** httpx (already dep), respx (test), Modal, vLLM, Helm, Terraform/GCP

**Design doc:** `docs/plans/2026-03-30-infra-sprint-design.md`

---

## Task 1: SelfHostedProvider — Factory + Config (commit 1, part 1)

**Files:**
- Modify: `agent_bench/core/provider.py:567-579` (add factory branch)
- Create: `configs/selfhosted_local.yaml`
- Create: `configs/selfhosted_modal.yaml`
- Test: `tests/test_selfhosted_provider.py`

### Step 1: Write failing test — factory creates SelfHostedProvider

```python
# tests/test_selfhosted_provider.py
"""Tests for the SelfHostedProvider (OpenAI-compatible endpoint)."""

import json

import httpx
import pytest
import respx

from agent_bench.core.config import AppConfig, ProviderConfig
from agent_bench.core.provider import create_provider
from agent_bench.core.types import Message, Role, ToolDefinition


class TestSelfHostedFactory:
    def test_factory_creates_selfhosted_provider(self, monkeypatch):
        """Factory returns SelfHostedProvider for 'selfhosted' config."""
        monkeypatch.setenv("MODAL_VLLM_URL", "http://fake:8000/v1")
        from agent_bench.core.provider import SelfHostedProvider

        config = AppConfig(provider=ProviderConfig(default="selfhosted"))
        provider = create_provider(config)
        assert isinstance(provider, SelfHostedProvider)

    def test_factory_raises_for_unknown_provider(self):
        config = AppConfig(provider=ProviderConfig(default="nonexistent"))
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider(config)
```

### Step 2: Run test to verify it fails

```bash
python -m pytest tests/test_selfhosted_provider.py::TestSelfHostedFactory::test_factory_creates_selfhosted_provider -v
```

Expected: `ImportError` — `SelfHostedProvider` does not exist yet.

### Step 3: Write SelfHostedProvider skeleton + register in factory

Add to `agent_bench/core/provider.py` (before `create_provider`, after `AnthropicProvider`):

```python
class SelfHostedProvider(LLMProvider):
    """Provider targeting any OpenAI-compatible endpoint (vLLM, TGI, Ollama).

    Reads base URL from config or MODAL_VLLM_URL env var.
    Reads auth token from config or MODAL_AUTH_TOKEN env var.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        import os

        self.config = config or load_config()
        self.base_url = os.environ.get("MODAL_VLLM_URL", "http://localhost:8000/v1")
        self.model = os.environ.get(
            "SELFHOSTED_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"
        )
        api_key = os.environ.get("MODAL_AUTH_TOKEN", "")
        self._supports_tool_calling: bool | None = None  # detected lazily

        model_pricing = self.config.provider.models.get(self.model)
        self._input_cost = model_pricing.input_cost_per_mtok if model_pricing else 0.0
        self._output_cost = model_pricing.output_cost_per_mtok if model_pricing else 0.0

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=120.0,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResponse:
        raise NotImplementedError("TODO")

    async def stream_complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        raise NotImplementedError("TODO")
        yield ""  # pragma: no cover

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return format_tools_openai(tools)
```

Update `create_provider` (line ~575):

```python
    elif name == "selfhosted":
        return SelfHostedProvider(config)
```

### Step 4: Run test to verify it passes

```bash
python -m pytest tests/test_selfhosted_provider.py::TestSelfHostedFactory -v
```

Expected: PASS (both tests).

---

## Task 2: SelfHostedProvider — complete() (commit 1, part 2)

**Files:**
- Modify: `agent_bench/core/provider.py` (implement `complete()`)
- Test: `tests/test_selfhosted_provider.py`

### Step 5: Write failing test — complete() with mocked response

Add to `tests/test_selfhosted_provider.py`:

```python
class TestSelfHostedComplete:
    @pytest.fixture
    def provider(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", "http://fake-vllm:8000/v1")
        from agent_bench.core.provider import SelfHostedProvider

        config = AppConfig(provider=ProviderConfig(default="selfhosted"))
        return SelfHostedProvider(config)

    @pytest.mark.asyncio
    async def test_complete_parses_response(self, provider):
        """SelfHostedProvider.complete() parses OpenAI-format response."""
        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Path params use curly braces. [source: fastapi.md]",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
        }

        with respx.mock:
            respx.post("http://fake-vllm:8000/v1/chat/completions").mock(
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
        """SelfHostedProvider.complete() parses tool_calls from response."""
        mock_response = {
            "id": "chatcmpl-test2",
            "object": "chat.completion",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "search_documents",
                                    "arguments": json.dumps({"query": "path params"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 60, "completion_tokens": 15, "total_tokens": 75},
        }
        tools = [
            ToolDefinition(
                name="search_documents",
                description="Search docs",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]

        with respx.mock:
            respx.post("http://fake-vllm:8000/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=mock_response)
            )
            response = await provider.complete(
                [Message(role=Role.USER, content="search for path params")],
                tools=tools,
            )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_abc"
        assert response.tool_calls[0].name == "search_documents"
        assert response.tool_calls[0].arguments == {"query": "path params"}

    @pytest.mark.asyncio
    async def test_complete_handles_malformed_tool_args(self, provider):
        """Malformed JSON in tool arguments falls back to empty dict."""
        mock_response = {
            "id": "chatcmpl-bad",
            "object": "chat.completion",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {
                                    "name": "search_documents",
                                    "arguments": "not valid json{{{",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        }

        with respx.mock:
            respx.post("http://fake-vllm:8000/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=mock_response)
            )
            response = await provider.complete(
                [Message(role=Role.USER, content="test")]
            )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].arguments == {}
```

### Step 6: Run tests to verify they fail

```bash
python -m pytest tests/test_selfhosted_provider.py::TestSelfHostedComplete -v
```

Expected: FAIL with `NotImplementedError`.

### Step 7: Implement complete()

Replace the `complete()` stub in `SelfHostedProvider`:

```python
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResponse:
        formatted_messages = format_messages_openai(messages)
        payload: dict = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
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
                resp.raise_for_status()
                break
            except httpx.TimeoutException as e:
                raise ProviderTimeoutError(f"Self-hosted timed out: {e}") from e

        latency_ms = (time.perf_counter() - start) * 1000
        data = resp.json()

        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        tool_calls: list[ToolCall] = []

        if choice["message"].get("tool_calls"):
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
```

Add `import httpx` at the top of `provider.py` (with the other imports).

### Step 8: Run tests to verify they pass

```bash
python -m pytest tests/test_selfhosted_provider.py::TestSelfHostedComplete -v
```

Expected: PASS (all 3 tests).

---

## Task 3: SelfHostedProvider — Retry, Timeout, Env Vars (commit 1, part 3)

**Files:**
- Modify: `agent_bench/core/provider.py`
- Test: `tests/test_selfhosted_provider.py`

### Step 9: Write failing tests — retry, timeout, env var fallback

Add to `tests/test_selfhosted_provider.py`:

```python
from agent_bench.core.provider import ProviderRateLimitError, ProviderTimeoutError


class TestSelfHostedRetryAndTimeout:
    @pytest.fixture
    def provider(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", "http://fake-vllm:8000/v1")
        from agent_bench.core.provider import SelfHostedProvider

        config = AppConfig(
            provider=ProviderConfig(default="selfhosted"),
            retry=RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.05),
        )
        return SelfHostedProvider(config)

    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self, provider):
        """Provider retries on 429 and succeeds on next attempt."""
        success_body = {
            "id": "ok",
            "object": "chat.completion",
            "model": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json=success_body)

        with respx.mock:
            respx.post("http://fake-vllm:8000/v1/chat/completions").mock(
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
            respx.post("http://fake-vllm:8000/v1/chat/completions").mock(
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
            respx.post("http://fake-vllm:8000/v1/chat/completions").mock(
                side_effect=httpx.ReadTimeout("timed out")
            )
            with pytest.raises(ProviderTimeoutError, match="timed out"):
                await provider.complete(
                    [Message(role=Role.USER, content="test")]
                )


class TestSelfHostedEnvVars:
    def test_reads_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", "http://my-modal-url:8000/v1")
        from agent_bench.core.provider import SelfHostedProvider

        config = AppConfig(provider=ProviderConfig(default="selfhosted"))
        provider = SelfHostedProvider(config)
        assert provider.base_url == "http://my-modal-url:8000/v1"

    def test_reads_auth_token_from_env(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", "http://fake:8000/v1")
        monkeypatch.setenv("MODAL_AUTH_TOKEN", "secret-token-123")
        from agent_bench.core.provider import SelfHostedProvider

        config = AppConfig(provider=ProviderConfig(default="selfhosted"))
        provider = SelfHostedProvider(config)
        assert provider.client.headers.get("authorization") == "Bearer secret-token-123"

    def test_no_auth_header_when_no_token(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", "http://fake:8000/v1")
        monkeypatch.delenv("MODAL_AUTH_TOKEN", raising=False)
        from agent_bench.core.provider import SelfHostedProvider

        config = AppConfig(provider=ProviderConfig(default="selfhosted"))
        provider = SelfHostedProvider(config)
        assert "authorization" not in {
            k.lower() for k in provider.client.headers.keys()
        }
```

Add this import at the top of the test file:

```python
from agent_bench.core.config import RetryConfig
```

### Step 10: Run tests to verify they pass

```bash
python -m pytest tests/test_selfhosted_provider.py -v
```

Expected: PASS (all 9 tests). The retry/timeout logic is already in the `complete()` from Step 7.

---

## Task 4: SelfHostedProvider — stream_complete() (commit 1, part 4)

**Files:**
- Modify: `agent_bench/core/provider.py`
- Test: `tests/test_selfhosted_provider.py`

### Step 11: Write failing test — stream_complete()

Add to `tests/test_selfhosted_provider.py`:

```python
class TestSelfHostedStream:
    @pytest.fixture
    def provider(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", "http://fake-vllm:8000/v1")
        from agent_bench.core.provider import SelfHostedProvider

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
            respx.post("http://fake-vllm:8000/v1/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    content=sse_body.encode(),
                    headers={"content-type": "text/event-stream"},
                )
            )
            chunks = []
            async for chunk in provider.stream_complete(
                [Message(role=Role.USER, content="Hi")]
            ):
                chunks.append(chunk)

        assert chunks == ["Hello ", "world"]
```

### Step 12: Run test to verify it fails

```bash
python -m pytest tests/test_selfhosted_provider.py::TestSelfHostedStream -v
```

Expected: FAIL with `NotImplementedError`.

### Step 13: Implement stream_complete()

Replace the `stream_complete()` stub in `SelfHostedProvider`:

```python
    async def stream_complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        formatted_messages = format_messages_openai(messages)
        payload: dict = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = self.format_tools(tools)
            payload["tool_choice"] = "auto"

        retry_cfg = self.config.retry
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
                        "selfhosted_stream_retry",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except httpx.TimeoutException as e:
                raise ProviderTimeoutError(f"Self-hosted timed out: {e}") from e

        for line in resp.text.split("\n"):
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str == "[DONE]":
                break
            try:
                chunk_data = json.loads(data_str)
                delta = chunk_data["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield delta["content"]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
```

### Step 14: Run tests to verify they pass

```bash
python -m pytest tests/test_selfhosted_provider.py -v
```

Expected: PASS (all 10 tests).

---

## Task 5: Config files + format_tools test + lint (commit 1, part 5)

**Files:**
- Create: `configs/selfhosted_local.yaml`
- Create: `configs/selfhosted_modal.yaml`
- Test: `tests/test_selfhosted_provider.py`

### Step 15: Create config files

**`configs/selfhosted_local.yaml`:**

```yaml
agent:
  max_iterations: 3
  temperature: 0.0

provider:
  default: selfhosted
  models:
    mistralai/Mistral-7B-Instruct-v0.3:
      input_cost_per_mtok: 0.0
      output_cost_per_mtok: 0.0
    gpt-4o-mini:
      input_cost_per_mtok: 0.15
      output_cost_per_mtok: 0.60

rag:
  chunking:
    strategy: recursive
    chunk_size: 512
    chunk_overlap: 64
  retrieval:
    strategy: hybrid
    rrf_k: 60
    candidates_per_system: 10
    top_k: 5
  reranker:
    enabled: true
    model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
    top_k: 5
  refusal_threshold: 0.02
  store_path: .cache/store

embedding:
  model: all-MiniLM-L6-v2
  cache_dir: .cache/embeddings

retry:
  max_retries: 3
  base_delay: 1.0
  max_delay: 8.0

memory:
  enabled: false

serving:
  host: 0.0.0.0
  port: 8000
  request_timeout_seconds: 120
  rate_limit_rpm: 10

evaluation:
  judge_provider: openai
  golden_dataset: agent_bench/evaluation/datasets/tech_docs_golden.json
```

**`configs/selfhosted_modal.yaml`:** Same as above (identical file). The difference is that `selfhosted_modal` will read `MODAL_VLLM_URL` env var at runtime, while `selfhosted_local` expects `http://localhost:8000/v1` from the Docker Compose vLLM service. Both use the same config structure.

### Step 16: Write test for format_tools and config loading

Add to `tests/test_selfhosted_provider.py`:

```python
class TestSelfHostedFormatTools:
    def test_format_tools_uses_openai_schema(self, monkeypatch):
        monkeypatch.setenv("MODAL_VLLM_URL", "http://fake:8000/v1")
        from agent_bench.core.provider import SelfHostedProvider

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
```

### Step 17: Run full test suite + lint

```bash
python -m pytest tests/test_selfhosted_provider.py -v
python -m pytest tests/ -v --tb=short
ruff check agent_bench/ tests/
ruff format agent_bench/ tests/
mypy agent_bench/ --ignore-missing-imports
```

Expected: All pass. 11 new tests, 0 regressions.

### Step 18: Commit

```bash
git add agent_bench/core/provider.py tests/test_selfhosted_provider.py configs/selfhosted_local.yaml configs/selfhosted_modal.yaml
git commit -m "feat: add SelfHostedProvider for OpenAI-compatible endpoints (vLLM, TGI, Ollama)"
```

---

## Task 6: Modal vLLM Deployment Scripts (commit 2)

**Files:**
- Create: `modal/__init__.py` (empty)
- Create: `modal/common.py`
- Create: `modal/serve_vllm.py`

### Step 19: Create modal/common.py

```python
"""Shared constants for Modal deployments."""

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
GPU_TYPE = "a10g"
VLLM_MAX_MODEL_LEN = 4096
VLLM_DTYPE = "half"
VLLM_GPU_MEMORY_UTILIZATION = 0.85

# Cost tracking (for provider comparison report)
# Modal A10G: ~$0.000361/sec (~$1.30/hr)
MODAL_A10G_COST_PER_SEC = 0.000361
```

### Step 20: Create modal/serve_vllm.py

Check Modal's current vLLM example before writing. The pattern changes between vLLM versions. Key contract: the deployed endpoint must expose `/v1/chat/completions` and `/health`.

```python
"""Deploy vLLM on Modal as an OpenAI-compatible endpoint.

Usage:
    modal deploy modal/serve_vllm.py     # Deploy (stays running, prints URL)
    modal serve modal/serve_vllm.py      # Dev mode (auto-redeploys on change)

The printed URL is the MODAL_VLLM_URL for SelfHostedProvider:
    export MODAL_VLLM_URL=https://<your-workspace>--agent-bench-vllm-serve.modal.run/v1
"""

import modal

from common import (
    MODEL_NAME,
    VLLM_DTYPE,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_MODEL_LEN,
)

MODELS_DIR = "/models"

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.6.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("agent-bench-vllm")
model_volume = modal.Volume.from_name("vllm-model-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu=modal.gpu.A10G(),
    container_idle_timeout=300,
    timeout=600,
    volumes={MODELS_DIR: model_volume},
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def serve():
    """Serve vLLM with OpenAI-compatible API."""
    from vllm.entrypoints.openai.api_server import build_app

    return build_app(
        model=MODEL_NAME,
        download_dir=MODELS_DIR,
        dtype=VLLM_DTYPE,
        max_model_len=VLLM_MAX_MODEL_LEN,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    )
```

**Implementation note:** The `build_app` call above is a sketch. At implementation time:
1. Run `modal deploy --help` to verify CLI syntax
2. Check `vllm.entrypoints.openai.api_server` for the current API — it may use `build_async_engine_client` + `init_app_state` instead of a single `build_app` call
3. Check Modal's vLLM example for the canonical pattern (may use `@modal.cls` instead of `@modal.asgi_app`)
4. Adapt to match both. Test with `modal serve modal/serve_vllm.py` before committing

### Step 21: Commit

```bash
git add modal/
git commit -m "feat: add Modal vLLM deployment scripts for serverless GPU inference"
```

---

## Task 7: Docker Compose vLLM (commit 3)

**Files:**
- Create: `docker/docker-compose.vllm.yml`

### Step 22: Create docker-compose.vllm.yml

```yaml
# docker/docker-compose.vllm.yml
#
# Local GPU serving via vLLM + agent-bench API.
# Requires: nvidia-container-toolkit
# See modal/serve_vllm.py for serverless alternative.
#
# Usage:
#   docker compose -f docker/docker-compose.vllm.yml up --build

services:
  vllm:
    image: vllm/vllm-openai:latest
    command:
      - --model=mistralai/Mistral-7B-Instruct-v0.3
      - --max-model-len=4096
      - --dtype=half
      - --gpu-memory-utilization=0.85
      - --host=0.0.0.0
      - --port=8000
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - vllm-cache:/root/.cache/huggingface
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s

  app:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    environment:
      - MODAL_VLLM_URL=http://vllm:8000/v1
      - AGENT_BENCH_ENV=selfhosted_local
    depends_on:
      vllm:
        condition: service_healthy
    ports:
      - "8080:7860"

volumes:
  vllm-cache:
```

### Step 23: Commit

```bash
git add docker/docker-compose.vllm.yml
git commit -m "feat: add Docker Compose config for local vLLM + API serving"
```

---

## Task 8: Benchmark Runner (commit 4)

**Files:**
- Create: `modal/run_benchmark.py`
- Create: `docs/provider_comparison.md` (generated after running)

### Step 24: Create modal/run_benchmark.py

```python
"""Run the 27-question benchmark against all provider configurations.

Usage:
    # Local: run against a deployed Modal endpoint
    python modal/run_benchmark.py --base-url https://...modal.run/v1

    # Or run entirely on Modal (mounts local repo)
    modal run modal/run_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_eval(config_path: str, env: dict[str, str]) -> dict:
    """Run scripts/evaluate.py and parse the JSON output."""
    output_path = f".cache/eval_{Path(config_path).stem}.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--config",
            config_path,
            "--mode",
            "deterministic",
            "--output",
            output_path,
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if result.returncode != 0:
        print(f"FAILED: {config_path}\n{result.stderr}", file=sys.stderr)
        return {"error": result.stderr}
    with open(Path(__file__).resolve().parent.parent / output_path) as f:
        return json.load(f)


def generate_report(all_results: dict[str, dict], output_path: str) -> None:
    """Generate docs/provider_comparison.md from benchmark results."""
    lines = [
        "# Provider Comparison: API vs Self-Hosted",
        "",
        "Benchmark: 27-question golden dataset (19 retrieval, 3 calculation, 5 out-of-scope).",
        "",
        "| Provider | Model | P@5 | R@5 | Citation Acc | Latency p50 (ms) | Cost/query |",
        "|----------|-------|-----|-----|--------------|-------------------|------------|",
    ]
    for name, results in all_results.items():
        if "error" in results:
            lines.append(f"| {name} | - | ERROR | - | - | - | - |")
            continue
        # Extract aggregate metrics from results list
        # (implementation depends on evaluate.py output format)
        lines.append(f"| {name} | ... | ... | ... | ... | ... | ... |")

    lines.extend(["", "---", "", "Generated by `modal/run_benchmark.py`"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines))
    print(f"Report written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run provider comparison benchmark")
    parser.add_argument("--base-url", required=True, help="Modal vLLM endpoint URL")
    args = parser.parse_args()

    configs = [
        ("openai", "configs/default.yaml"),
        ("anthropic", "configs/anthropic.yaml"),
        ("selfhosted_modal", "configs/selfhosted_modal.yaml"),
    ]

    all_results = {}
    for name, config_path in configs:
        print(f"\n--- Running: {name} ({config_path}) ---")
        env = os.environ.copy()
        if name == "selfhosted_modal":
            env["MODAL_VLLM_URL"] = args.base_url
        all_results[name] = run_eval(config_path, env)

    generate_report(all_results, "docs/provider_comparison.md")


if __name__ == "__main__":
    main()
```

### Step 25: Commit

```bash
git add modal/run_benchmark.py
git commit -m "feat: add benchmark runner for provider comparison (API vs self-hosted)"
```

Note: `docs/provider_comparison.md` is committed separately after actually running the benchmark with real Modal endpoints and API keys. The runner script generates it.

---

## Task 9: Helm Chart (commit 5)

**Files:**
- Create: `k8s/helm/agent-bench/Chart.yaml`
- Create: `k8s/helm/agent-bench/values.yaml`
- Create: `k8s/helm/agent-bench/values-dev.yaml`
- Create: `k8s/helm/agent-bench/values-prod.yaml`
- Create: `k8s/helm/agent-bench/templates/_helpers.tpl`
- Create: `k8s/helm/agent-bench/templates/deployment.yaml`
- Create: `k8s/helm/agent-bench/templates/service.yaml`
- Create: `k8s/helm/agent-bench/templates/hpa.yaml`
- Create: `k8s/helm/agent-bench/templates/configmap.yaml`
- Create: `k8s/helm/agent-bench/templates/secret.yaml`

### Step 26: Create Chart.yaml

```yaml
apiVersion: v2
name: agent-bench
description: Agentic RAG system with self-hosted LLM support
type: application
version: 0.1.0
appVersion: "0.1.0"
```

### Step 27: Create values.yaml

```yaml
replicaCount: 2

image:
  repository: agent-bench
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8000

provider:
  type: selfhosted
  selfhosted:
    model: mistralai/Mistral-7B-Instruct-v0.3
    modalEndpoint: ""
    modalAuthToken: ""
  openaiApiKey: ""
  anthropicApiKey: ""

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 8
  targetCPUUtilization: 70

resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi

probes:
  liveness:
    path: /health
    initialDelaySeconds: 10
    periodSeconds: 30
  readiness:
    path: /health
    initialDelaySeconds: 5
    periodSeconds: 10
```

### Step 28: Create values-dev.yaml

```yaml
replicaCount: 1

autoscaling:
  enabled: false

resources:
  requests:
    cpu: 250m
    memory: 512Mi
  limits:
    cpu: 1000m
    memory: 2Gi
```

### Step 29: Create values-prod.yaml

```yaml
replicaCount: 3

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 8
  targetCPUUtilization: 70

resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

### Step 30: Create templates/_helpers.tpl

```yaml
{{/*
Expand the name of the chart.
*/}}
{{- define "agent-bench.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "agent-bench.fullname" -}}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "agent-bench.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{ include "agent-bench.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "agent-bench.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agent-bench.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
```

### Step 31: Create templates/deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "agent-bench.fullname" . }}
  labels:
    {{- include "agent-bench.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "agent-bench.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "agent-bench.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: api
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 7860
              protocol: TCP
          envFrom:
            - configMapRef:
                name: {{ include "agent-bench.fullname" . }}-config
            - secretRef:
                name: {{ include "agent-bench.fullname" . }}-secrets
          livenessProbe:
            httpGet:
              path: {{ .Values.probes.liveness.path }}
              port: 7860
            initialDelaySeconds: {{ .Values.probes.liveness.initialDelaySeconds }}
            periodSeconds: {{ .Values.probes.liveness.periodSeconds }}
          readinessProbe:
            httpGet:
              path: {{ .Values.probes.readiness.path }}
              port: 7860
            initialDelaySeconds: {{ .Values.probes.readiness.initialDelaySeconds }}
            periodSeconds: {{ .Values.probes.readiness.periodSeconds }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
```

**Note:** Container port is `7860` (matching the Dockerfile `EXPOSE 7860`). The Service maps this to `8000` externally.

### Step 32: Create templates/service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "agent-bench.fullname" . }}
  labels:
    {{- include "agent-bench.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: 7860
      protocol: TCP
      name: http
  selector:
    {{- include "agent-bench.selectorLabels" . | nindent 4 }}
```

### Step 33: Create templates/hpa.yaml

```yaml
{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "agent-bench.fullname" . }}
  labels:
    {{- include "agent-bench.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "agent-bench.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilization }}
{{- end }}
```

### Step 34: Create templates/configmap.yaml

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "agent-bench.fullname" . }}-config
  labels:
    {{- include "agent-bench.labels" . | nindent 4 }}
data:
  AGENT_BENCH_ENV: "selfhosted_modal"
  SELFHOSTED_MODEL: {{ .Values.provider.selfhosted.model | quote }}
```

### Step 35: Create templates/secret.yaml

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: {{ include "agent-bench.fullname" . }}-secrets
  labels:
    {{- include "agent-bench.labels" . | nindent 4 }}
type: Opaque
stringData:
  MODAL_VLLM_URL: {{ .Values.provider.selfhosted.modalEndpoint | quote }}
  MODAL_AUTH_TOKEN: {{ .Values.provider.selfhosted.modalAuthToken | quote }}
  OPENAI_API_KEY: {{ .Values.provider.openaiApiKey | quote }}
  ANTHROPIC_API_KEY: {{ .Values.provider.anthropicApiKey | quote }}
```

### Step 36: Validate Helm chart

```bash
helm lint k8s/helm/agent-bench/
helm template test-release k8s/helm/agent-bench/ -f k8s/helm/agent-bench/values-dev.yaml
helm template test-release k8s/helm/agent-bench/ -f k8s/helm/agent-bench/values-prod.yaml
```

Expected: No errors. Templates render correctly for both dev and prod values.

### Step 37: Commit

```bash
git add k8s/
git commit -m "feat: add Helm chart for K8s deployment with dev/prod values"
```

---

## Task 10: Terraform GKE Modules (commit 6)

**Files:**
- Create: `terraform/main.tf`
- Create: `terraform/variables.tf`
- Create: `terraform/outputs.tf`
- Create: `terraform/terraform.tfvars.example`
- Create: `terraform/modules/networking/main.tf`
- Create: `terraform/modules/networking/variables.tf`
- Create: `terraform/modules/gke/main.tf`
- Create: `terraform/modules/gke/variables.tf`
- Create: `terraform/modules/gke/outputs.tf`

### Step 38: Create terraform/variables.tf

```hcl
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for the cluster"
  type        = string
  default     = "europe-west1"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "agent-bench-cluster"
}
```

### Step 39: Create terraform/main.tf

```hcl
terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

module "networking" {
  source       = "./modules/networking"
  project_id   = var.project_id
  region       = var.region
  cluster_name = var.cluster_name
}

module "gke" {
  source           = "./modules/gke"
  project_id       = var.project_id
  region           = var.region
  cluster_name     = var.cluster_name
  network          = module.networking.network_name
  subnetwork       = module.networking.subnetwork_name
  cpu_node_count   = 2
  cpu_machine_type = "e2-standard-4"
}
```

### Step 40: Create terraform/outputs.tf

```hcl
output "cluster_name" {
  description = "GKE cluster name"
  value       = module.gke.cluster_name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = module.gke.cluster_endpoint
  sensitive   = true
}

output "kubeconfig_command" {
  description = "Command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${var.cluster_name} --region ${var.region} --project ${var.project_id}"
}
```

### Step 41: Create terraform/terraform.tfvars.example

```hcl
# Copy to terraform.tfvars and fill in values.
# terraform.tfvars is gitignored.

project_id   = "your-gcp-project-id"
region       = "europe-west1"
cluster_name = "agent-bench-cluster"
```

### Step 42: Create terraform/modules/networking/variables.tf

```hcl
variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "cluster_name" {
  type = string
}
```

### Step 43: Create terraform/modules/networking/main.tf

```hcl
resource "google_compute_network" "vpc" {
  name                    = "${var.cluster_name}-vpc"
  auto_create_subnetworks = false
  project                 = var.project_id
}

resource "google_compute_subnetwork" "subnet" {
  name          = "${var.cluster_name}-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id
  project       = var.project_id

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/20"
  }
}

resource "google_compute_firewall" "allow_internal" {
  name    = "${var.cluster_name}-allow-internal"
  network = google_compute_network.vpc.name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = ["10.0.0.0/8"]
}

resource "google_compute_firewall" "allow_health_checks" {
  name    = "${var.cluster_name}-allow-health-checks"
  network = google_compute_network.vpc.name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8000", "7860"]
  }

  # GCP health check IP ranges
  source_ranges = ["35.191.0.0/16", "130.211.0.0/22"]
}

output "network_name" {
  value = google_compute_network.vpc.name
}

output "subnetwork_name" {
  value = google_compute_subnetwork.subnet.name
}
```

### Step 44: Create terraform/modules/gke/variables.tf

```hcl
variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "cluster_name" {
  type = string
}

variable "network" {
  type = string
}

variable "subnetwork" {
  type = string
}

variable "cpu_node_count" {
  type    = number
  default = 2
}

variable "cpu_machine_type" {
  type    = string
  default = "e2-standard-4"
}
```

### Step 45: Create terraform/modules/gke/main.tf

```hcl
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.region
  project  = var.project_id

  network    = var.network
  subnetwork = var.subnetwork

  # Autopilot disabled — we manage node pools explicitly
  enable_autopilot = false

  # Remove default node pool (we create our own)
  remove_default_node_pool = true
  initial_node_count       = 1

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
}

resource "google_container_node_pool" "cpu_pool" {
  name       = "${var.cluster_name}-cpu-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = var.cpu_node_count
  project    = var.project_id

  node_config {
    machine_type = var.cpu_machine_type
    disk_size_gb = 50
    disk_type    = "pd-standard"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }
}
```

### Step 46: Create terraform/modules/gke/outputs.tf

```hcl
output "cluster_name" {
  value = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  value     = google_container_cluster.primary.endpoint
  sensitive = true
}
```

### Step 47: Add terraform.tfvars to .gitignore

Append to `.gitignore`:

```
terraform.tfvars
.terraform/
*.tfstate
*.tfstate.backup
```

### Step 48: Validate Terraform

```bash
cd terraform && terraform init -backend=false && terraform validate
```

Expected: `Success! The configuration is valid.`

### Step 49: Commit

```bash
git add terraform/ .gitignore
git commit -m "feat: add Terraform GKE modules for API cluster (CPU-only, GCP)"
```

---

## Task 11: Makefile + DECISIONS.md + README (commit 7)

**Files:**
- Modify: `Makefile`
- Modify: `DECISIONS.md`
- Modify: `README.md`

### Step 50: Add Makefile targets

Append to `Makefile`:

```makefile
## --- Infrastructure ---

modal-deploy:  ## Deploy vLLM on Modal (prints endpoint URL)
	modal deploy modal/serve_vllm.py

modal-stop:  ## Stop Modal deployment
	modal app stop agent-bench-vllm

vllm-up:  ## Start local vLLM via Docker Compose (requires NVIDIA GPU)
	docker compose -f docker/docker-compose.vllm.yml up --build

benchmark-all:  ## Run provider comparison (requires Modal deployment + API keys)
	$(PYTHON) modal/run_benchmark.py --base-url $(MODAL_VLLM_URL)

k8s-dev:  ## Deploy to minikube (dev values)
	helm install agent-bench k8s/helm/agent-bench/ -f k8s/helm/agent-bench/values-dev.yaml

k8s-prod:  ## Deploy via Helm (prod values)
	helm install agent-bench k8s/helm/agent-bench/ -f k8s/helm/agent-bench/values-prod.yaml

tf-plan:  ## Run terraform plan (no apply)
	cd terraform && terraform plan

tf-validate:  ## Validate terraform syntax
	cd terraform && terraform validate
```

### Step 51: Add DECISIONS.md entries

Append to `DECISIONS.md`:

```markdown

## Why vLLM over TGI / llama.cpp

vLLM has the widest model support, best throughput via PagedAttention, and a native
OpenAI-compatible server (`/v1/chat/completions`). TGI is a valid alternative; llama.cpp
targets different use cases (edge/CPU inference). This is a deliberate choice, not
ignorance of alternatives.

## Why Modal for GPU inference

Serverless GPU eliminates idle cost and GPU node management. A10G at ~$1.30/hr costs
~$0.50 per full 27-question benchmark run. The Docker Compose path (`docker-compose.vllm.yml`)
is retained for users who have local GPUs or prefer persistent serving.

## Why split topology (K8s API + Modal GPU)

The API layer (retrieval, orchestration, tool routing) is CPU-bound and benefits from
horizontal scaling via K8s HPA. The LLM inference layer is GPU-bound and benefits from
serverless elasticity — Modal scales to zero when idle, scales up on demand with no node
provisioning. Co-locating both in K8s would require GPU node pools with idle cost,
node autoscaler latency, and NVIDIA device plugin management. This mirrors a common
production pattern.

## Why Helm only, not Kustomize + Helm

Showing two K8s deployment methods for the same app adds complexity without demonstrating
distinct skills. Helm with `values-dev.yaml` / `values-prod.yaml` covers
environment-specific configuration cleanly.

## Why CPU-based HPA, not custom metrics

CPU utilization works without a Prometheus adapter or custom metrics server. A production
improvement would use the Prometheus adapter to scale on p95 latency from the `/metrics`
endpoint — this requires bridging the JSON metrics to Prometheus exposition format.
Documented as a follow-up.

## Why env var fallback in SelfHostedProvider

Follows the same pattern as OpenAIProvider reading `OPENAI_API_KEY`. The YAML config
provides defaults; env vars override at runtime. No config loader changes needed.

## Why startup smoke test for tool-call detection

Checking `/v1/models` metadata for tool-calling support is unreliable — model metadata
doesn't consistently report this capability. Instead, the provider sends one tool-calling
request at init and checks if the response contains `tool_calls`. The result is cached as
`self._supports_tool_calling`.
```

### Step 52: Update README.md

Add after the "With Docker" section:

```markdown
### Self-Hosted LLM via Modal (no local GPU needed)

```bash
# Deploy vLLM on Modal (A10G GPU, prints endpoint URL)
make modal-deploy

# Set the endpoint URL
export MODAL_VLLM_URL=https://your--agent-bench-vllm-serve.modal.run/v1

# Run with self-hosted provider
make serve CONFIG=configs/selfhosted_modal.yaml

# Run the full provider comparison benchmark
make benchmark-all
```

### Self-Hosted LLM via Docker Compose (requires local NVIDIA GPU)

```bash
docker compose -f docker/docker-compose.vllm.yml up --build
```

### Kubernetes (Helm)

```bash
# Dev (1 replica, no HPA)
make k8s-dev

# Prod (3 replicas, HPA enabled)
make k8s-prod
```

See `docs/k8s-local-setup.md` for minikube walkthrough.
```

Update the Architecture section to add the provider tree and infra diagram from the design doc.

Update the "Skills Demonstrated" section to add:
- **Infrastructure:** Kubernetes (Helm), Terraform (GCP/GKE), self-hosted LLM serving (vLLM)
- **MLOps:** Provider comparison benchmark (API vs self-hosted, real measured data)

### Step 53: Create docs/k8s-local-setup.md

```markdown
# Kubernetes Local Setup (minikube)

## Prerequisites

- [minikube](https://minikube.sigs.k8s.io/docs/start/)
- [Helm](https://helm.sh/docs/intro/install/)
- Docker

## Deploy

```bash
# Start minikube
minikube start --cpus=4 --memory=8192

# Build image inside minikube's Docker daemon
eval $(minikube docker-env)
docker build -t agent-bench:latest -f docker/Dockerfile .

# Deploy with dev values
helm install agent-bench k8s/helm/agent-bench/ \
  -f k8s/helm/agent-bench/values-dev.yaml \
  --set provider.selfhosted.modalEndpoint=$MODAL_VLLM_URL

# Verify
kubectl get pods
kubectl port-forward svc/agent-bench 8080:8000

# Test
curl http://localhost:8080/health
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I define a path parameter in FastAPI?"}'
```

## Teardown

```bash
helm uninstall agent-bench
minikube stop
```
```

### Step 54: Run full test suite

```bash
python -m pytest tests/ -v --tb=short
ruff check agent_bench/ tests/
mypy agent_bench/ --ignore-missing-imports
```

Expected: All pass, no regressions.

### Step 55: Commit

```bash
git add Makefile DECISIONS.md README.md docs/k8s-local-setup.md
git commit -m "docs: add infra documentation, Makefile targets, and architecture updates"
```

---

## Summary

| Commit | Task | Files | Tests |
|--------|------|-------|-------|
| 1 | SelfHostedProvider + configs | `provider.py`, `test_selfhosted_provider.py`, 2 YAML configs | 11 new |
| 2 | Modal vLLM scripts | `modal/common.py`, `modal/serve_vllm.py` | Manual deploy |
| 3 | Docker Compose vLLM | `docker/docker-compose.vllm.yml` | Declarative |
| 4 | Benchmark runner | `modal/run_benchmark.py` | Manual run |
| 5 | Helm chart | `k8s/helm/agent-bench/` (10 files) | `helm lint/template` |
| 6 | Terraform GKE | `terraform/` (9 files), `.gitignore` | `terraform validate` |
| 7 | Docs + Makefile | `Makefile`, `DECISIONS.md`, `README.md`, `k8s-local-setup.md` | Full suite |

**Total new tests:** 11 (in `tests/test_selfhosted_provider.py`)
**Total new files:** ~25
**No existing tests broken:** All changes are additive.
