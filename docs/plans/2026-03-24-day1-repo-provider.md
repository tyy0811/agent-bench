# Day 1: Repo Scaffolding + Provider Abstraction

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up the repository with installable package, CI, config system, and the full provider abstraction (OpenAI real + Mock + Anthropic stub) with tests.

**Architecture:** Pydantic v2 models for all types, YAML-based config loaded via pydantic-settings, async provider interface with three implementations. All tests deterministic via MockProvider — no API keys needed.

**Tech Stack:** Python 3.11, setuptools, pytest, pytest-asyncio, ruff, mypy, httpx, respx, openai SDK, anthropic SDK, pydantic v2, pyyaml, structlog

---

### Task 1: Project Skeleton + pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `agent_bench/__init__.py`
- Create: `agent_bench/core/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "agent-bench"
version = "0.1.0"
description = "Evaluation-first agentic RAG system built from API primitives"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "openai>=1.50.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    "pyyaml>=6.0",
    "sentence-transformers>=3.0.0",
    "faiss-cpu>=1.8.0",
    "rank-bm25>=0.2.2",
    "structlog>=24.0.0",
    "httpx>=0.27.0",
    "simpleeval>=1.0.0",
    "numpy>=1.26.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.6.0",
    "mypy>=1.11.0",
    "respx>=0.21.0",
]

[build-system]
requires = ["setuptools>=69.0"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
```

**Step 2: Create .gitignore**

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg
.cache/
.mypy_cache/
.pytest_cache/
.ruff_cache/
*.faiss
*.pkl
.env
.venv/
venv/
```

**Step 3: Create package init files**

`agent_bench/__init__.py`:
```python
"""Evaluation-first agentic RAG system built from API primitives."""
```

`agent_bench/core/__init__.py`:
```python
"""Core types, configuration, and provider abstraction."""
```

`tests/__init__.py`: empty file.

**Step 4: Install the package**

Run: `pip install -e ".[dev]"`
Expected: Successful installation with all dependencies.

**Step 5: Verify install**

Run: `python -c "import agent_bench; print('ok')"`
Expected: `ok`

**Step 6: Commit**

```bash
git add pyproject.toml .gitignore agent_bench/__init__.py agent_bench/core/__init__.py tests/__init__.py
git commit -m "feat: initialize project skeleton with pyproject.toml"
```

---

### Task 2: Makefile + CI

**Files:**
- Create: `Makefile`
- Create: `.github/workflows/ci.yaml`

**Step 1: Create Makefile**

```makefile
.PHONY: install test lint serve ingest evaluate-fast evaluate-full benchmark docker

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check agent_bench/ tests/
	ruff format --check agent_bench/ tests/
	mypy agent_bench/ --ignore-missing-imports

serve:
	uvicorn agent_bench.serving.app:create_app --factory --reload --port 8000

ingest:
	python scripts/ingest.py --config configs/tasks/tech_docs.yaml

evaluate-fast:
	python scripts/evaluate.py --config configs/default.yaml --mode deterministic

evaluate-full:
	python scripts/evaluate.py --config configs/default.yaml --mode full

benchmark:
	python scripts/benchmark.py --output docs/benchmark_report.md

docker:
	docker-compose -f docker/docker-compose.yaml up --build
```

**Step 2: Create CI workflow**

`.github/workflows/ci.yaml`:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: make lint
      - run: make test
```

**Step 3: Verify Makefile**

Run: `make test`
Expected: `no tests ran` (0 tests collected, no failures — we haven't written tests yet)

**Step 4: Commit**

```bash
git add Makefile .github/workflows/ci.yaml
git commit -m "feat: add Makefile and GitHub Actions CI workflow"
```

---

### Task 3: Shared Types (`core/types.py`)

**Files:**
- Create: `agent_bench/core/types.py`

**Step 1: Write the test** (in `tests/test_provider.py` — we'll add to this file throughout)

Create `tests/test_provider.py`:
```python
"""Tests for core types and provider abstraction."""

import pytest

from agent_bench.core.types import (
    CompletionResponse,
    Message,
    Role,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_provider.py::TestCoreTypes -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_bench.core.types'`

**Step 3: Write the implementation**

`agent_bench/core/types.py`:
```python
"""Shared type definitions used across agent-bench."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class Message(BaseModel):
    role: Role
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict  # JSON Schema


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class CompletionResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage
    provider: str
    model: str
    latency_ms: float
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_provider.py::TestCoreTypes -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add agent_bench/core/types.py tests/test_provider.py
git commit -m "feat: add shared type definitions (Message, ToolCall, TokenUsage, etc.)"
```

---

### Task 4: Configuration (`core/config.py` + YAML files)

**Files:**
- Create: `agent_bench/core/config.py`
- Create: `configs/default.yaml`
- Create: `configs/tasks/tech_docs.yaml`

**Step 1: Write the test**

Append to `tests/test_provider.py`:
```python
from agent_bench.core.config import load_config, AppConfig


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
        from agent_bench.core.config import load_task_config

        task = load_task_config("tech_docs")
        assert task.name == "tech_docs"
        assert "search_documents" in task.system_prompt
        assert "[source:" in task.system_prompt
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_provider.py::TestConfig -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Create configs/default.yaml**

```yaml
agent:
  max_iterations: 3
  temperature: 0.0

provider:
  default: openai
  models:
    gpt-4o-mini:
      input_cost_per_mtok: 0.15
      output_cost_per_mtok: 0.60
    claude-sonnet-4-20250514:
      input_cost_per_mtok: 3.0
      output_cost_per_mtok: 15.0

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
    enabled: false
  store_path: .cache/store

embedding:
  model: all-MiniLM-L6-v2
  cache_dir: .cache/embeddings

serving:
  host: 0.0.0.0
  port: 8000
  request_timeout_seconds: 30

evaluation:
  judge_provider: openai
  golden_dataset: agent_bench/evaluation/datasets/tech_docs_golden.json
```

**Step 4: Create configs/tasks/tech_docs.yaml**

```yaml
task:
  name: tech_docs
  description: "Q&A over technical documentation"
  system_prompt: |
    You are a technical documentation assistant. You have access to tools
    that let you search a documentation corpus and perform calculations.

    Rules:
    - Use search_documents to find relevant information before answering.
    - Base your answer ONLY on the retrieved documents.
    - Cite sources inline as [source: filename.md] for each claim.
    - If the documents don't contain the answer, respond with:
      "The documentation does not contain information about this topic."
    - Use calculator for any numerical computations.
    - Be concise and precise.
  document_dir: data/tech_docs/
```

**Step 5: Write the implementation**

`agent_bench/core/config.py`:
```python
"""Configuration loading from YAML files via Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


# --- Nested config models ---


class AgentConfig(BaseModel):
    max_iterations: int = 3
    temperature: float = 0.0


class ModelPricing(BaseModel):
    input_cost_per_mtok: float
    output_cost_per_mtok: float


class ProviderConfig(BaseModel):
    default: str = "openai"
    models: dict[str, ModelPricing] = {}


class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64


class RetrievalConfig(BaseModel):
    strategy: str = "hybrid"
    rrf_k: int = 60
    candidates_per_system: int = 10
    top_k: int = 5


class RerankerConfig(BaseModel):
    enabled: bool = False


class RAGConfig(BaseModel):
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()
    store_path: str = ".cache/store"


class EmbeddingConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    cache_dir: str = ".cache/embeddings"


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout_seconds: int = 30


class EvaluationConfig(BaseModel):
    judge_provider: str = "openai"
    golden_dataset: str = "agent_bench/evaluation/datasets/tech_docs_golden.json"


class AppConfig(BaseModel):
    agent: AgentConfig = AgentConfig()
    provider: ProviderConfig = ProviderConfig()
    rag: RAGConfig = RAGConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    serving: ServingConfig = ServingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


# --- Task config ---


class TaskConfig(BaseModel):
    name: str
    description: str
    system_prompt: str
    document_dir: str = "data/tech_docs/"


class TaskFileConfig(BaseModel):
    task: TaskConfig


# --- Loaders ---

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_config(path: Path | None = None) -> AppConfig:
    """Load application config from YAML."""
    if path is None:
        path = _CONFIG_DIR / "default.yaml"
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return AppConfig.model_validate(data)


def load_task_config(task_name: str, path: Path | None = None) -> TaskConfig:
    """Load a task-specific config from YAML."""
    if path is None:
        path = _CONFIG_DIR / "tasks" / f"{task_name}.yaml"
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return TaskFileConfig.model_validate(data).task
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_provider.py::TestConfig -v`
Expected: 4 passed

**Step 7: Commit**

```bash
git add agent_bench/core/config.py configs/default.yaml configs/tasks/tech_docs.yaml
git commit -m "feat: add config system with Pydantic models and YAML loading"
```

---

### Task 5: Provider Interface + MockProvider

**Files:**
- Create: `agent_bench/core/provider.py`
- Modify: `tests/test_provider.py`
- Modify: `tests/conftest.py`

**Step 1: Write the tests**

Create `tests/conftest.py`:
```python
"""Shared test fixtures."""

import pytest


@pytest.fixture
def mock_provider():
    """MockProvider instance for deterministic testing."""
    from agent_bench.core.provider import MockProvider

    return MockProvider()
```

Append to `tests/test_provider.py`:
```python
from agent_bench.core.provider import (
    LLMProvider,
    MockProvider,
    OpenAIProvider,
    AnthropicProvider,
    create_provider,
    ProviderTimeoutError,
)


class TestMockProvider:
    @pytest.mark.asyncio
    async def test_returns_tool_calls_on_first_call(self, mock_provider):
        """First call (no tool results in messages) returns tool_calls."""
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
        """When messages contain tool results, return final answer (no tool_calls)."""
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Search for FastAPI path params"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[ToolCall(id="call_1", name="search_documents", arguments={"query": "path params"})],
            ),
            Message(role=Role.TOOL, content="Path params use curly braces.", tool_call_id="call_1"),
        ]
        response = await mock_provider.complete(messages)
        assert response.tool_calls == []
        assert len(response.content) > 0
        assert response.usage.input_tokens > 0

    @pytest.mark.asyncio
    async def test_returns_answer_without_tools(self, mock_provider):
        """When no tools provided, return a direct answer."""
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_provider.py::TestMockProvider -v`
Expected: FAIL — `ImportError`

**Step 3: Write the implementation**

`agent_bench/core/provider.py`:
```python
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


class MockProvider(LLMProvider):
    """Deterministic provider for testing.

    Behavior:
    - If tools are provided AND no Role.TOOL messages exist → returns tool_calls
    - If Role.TOOL messages exist OR no tools → returns final text answer
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
            # First call: simulate tool use
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

        # Final answer
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


class OpenAIProvider(LLMProvider):
    """OpenAI API provider using gpt-4o-mini."""

    def __init__(self, config: AppConfig | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install openai") from e

        self.config = config or load_config()
        self.client = AsyncOpenAI()
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

        formatted_messages = self._format_messages(messages)
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
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
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
            provider="openai",
            model=self.model,
            latency_ms=latency_ms,
        )

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
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

    def _format_messages(self, messages: list[Message]) -> list[dict]:
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


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider — stub for V2."""

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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_provider.py::TestMockProvider -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add agent_bench/core/provider.py tests/conftest.py tests/test_provider.py
git commit -m "feat: add provider abstraction with MockProvider, OpenAI, and Anthropic stub"
```

---

### Task 6: OpenAI Provider Tests (no API call) + Anthropic Stub Test

**Files:**
- Modify: `tests/test_provider.py`

**Step 1: Write the tests**

Append to `tests/test_provider.py`:
```python
class TestOpenAIProvider:
    def test_format_tools_produces_openai_schema(self):
        """format_tools() produces correct OpenAI function-calling schema — no API call."""
        provider = OpenAIProvider.__new__(OpenAIProvider)
        # Bypass __init__ to avoid needing API key — format_tools is pure
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
        formatted = provider.format_tools(tools)
        assert len(formatted) == 1
        assert formatted[0]["type"] == "function"
        func = formatted[0]["function"]
        assert func["name"] == "search_documents"
        assert func["description"] == "Search the documentation corpus"
        assert func["parameters"]["required"] == ["query"]

    def test_format_messages_maps_roles(self):
        """Message formatting maps internal roles to OpenAI role strings."""
        provider = OpenAIProvider.__new__(OpenAIProvider)
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
        formatted = provider._format_messages(messages)
        assert formatted[0]["role"] == "system"
        assert formatted[1]["role"] == "user"
        assert formatted[2]["role"] == "assistant"
        assert formatted[2]["tool_calls"][0]["id"] == "call_1"
        assert formatted[2]["tool_calls"][0]["function"]["name"] == "search"
        assert formatted[3]["role"] == "tool"
        assert formatted[3]["tool_call_id"] == "call_1"


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


class TestProviderFactory:
    def test_create_mock_provider(self):
        from agent_bench.core.config import AppConfig, ProviderConfig

        config = AppConfig(provider=ProviderConfig(default="mock"))
        provider = create_provider(config)
        assert isinstance(provider, MockProvider)

    def test_create_unknown_provider_raises(self):
        from agent_bench.core.config import AppConfig, ProviderConfig

        config = AppConfig(provider=ProviderConfig(default="unknown"))
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider(config)
```

**Step 2: Run all tests**

Run: `pytest tests/test_provider.py -v`
Expected: 15 passed (5 types + 4 config + 4 mock + 4 openai/anthropic/factory)

**Step 3: Commit**

```bash
git add tests/test_provider.py
git commit -m "test: add OpenAI format tests, Anthropic stub tests, provider factory tests"
```

---

### Task 7: Lint + Final Gate

**Step 1: Run the linter**

Run: `make lint`
Expected: May have formatting issues.

**Step 2: Fix any lint issues**

Run: `ruff format agent_bench/ tests/`
Then: `ruff check --fix agent_bench/ tests/`

**Step 3: Run full test suite**

Run: `make test`
Expected: 15 passed

**Step 4: Verify the Day 1 gate**

Run: `make install && make test`
Expected: Install succeeds, 15 tests pass.

**Step 5: Commit any lint fixes**

```bash
git add -A
git commit -m "style: fix lint and formatting issues"
```

---

## Summary

**7 tasks, 15 tests, 7 files created:**

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package definition with correct `setuptools.build_meta` backend |
| `.gitignore` | Standard Python ignores |
| `Makefile` | Build/test/serve commands |
| `.github/workflows/ci.yaml` | GitHub Actions CI |
| `agent_bench/core/types.py` | Message, ToolCall, TokenUsage, CompletionResponse, ToolDefinition |
| `agent_bench/core/config.py` | AppConfig, TaskConfig, YAML loaders |
| `agent_bench/core/provider.py` | LLMProvider ABC, MockProvider, OpenAIProvider, AnthropicProvider stub |
| `configs/default.yaml` | Default app config with OpenAI pricing |
| `configs/tasks/tech_docs.yaml` | Tech docs task with citation-aware system prompt |
| `tests/conftest.py` | mock_provider fixture |
| `tests/test_provider.py` | 15 tests across types, config, mock, openai format, anthropic stub, factory |

**Day 1 gate:** `make install && make test` — 15 tests green, zero API keys needed.
