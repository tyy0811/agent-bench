# LangChain Baseline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a LangChain tool-calling agent that runs the same 27-question golden dataset with the same metrics, producing a side-by-side comparison against the custom pipeline.

**Architecture:** A new `agent_bench/langchain_baseline/` module wraps the existing async `Retriever` and tools as LangChain `BaseRetriever` / `StructuredTool` objects, feeds them into a `create_tool_calling_agent` executor, and runs the golden dataset through a runner that produces `EvalResult` objects identical to the existing harness. The search tool captures retrieval metadata via a stateful wrapper so metrics like P@5, R@5, and citation accuracy can be computed using the exact same functions in `agent_bench/evaluation/metrics.py`.

**Tech Stack:** `langchain>=0.2`, `langchain-openai>=0.1`, `langchain-anthropic>=0.1`, existing `agent_bench` infrastructure.

---

## Task 1: Add LangChain Dependencies

**Files:**
- Modify: `pyproject.toml:6-21`

**Step 1: Add dependencies to pyproject.toml**

Add these 3 packages to the `dependencies` list (after the existing `simpleeval` line):

```toml
    "langchain>=0.2.0",
    "langchain-openai>=0.1.0",
    "langchain-anthropic>=0.1.0",
```

**Step 2: Install and verify imports**

Run: `pip install -e ".[dev]"`

Then verify:

Run: `python -c "from langchain.agents import create_tool_calling_agent, AgentExecutor; from langchain_openai import ChatOpenAI; from langchain_anthropic import ChatAnthropic; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add langchain dependencies for baseline comparison"
```

---

## Task 2: Retriever Wrapper

**Files:**
- Create: `agent_bench/langchain_baseline/__init__.py`
- Create: `agent_bench/langchain_baseline/retriever.py`
- Create: `tests/test_langchain_baseline/__init__.py`
- Create: `tests/test_langchain_baseline/test_retriever.py`

**Step 1: Create module skeleton**

Create `agent_bench/langchain_baseline/__init__.py`:

```python
"""LangChain baseline: tool-calling agent for framework comparison."""
```

Create `tests/test_langchain_baseline/__init__.py`:

```python
```

**Step 2: Write the failing test**

Create `tests/test_langchain_baseline/test_retriever.py`:

```python
"""Tests for LangChain retriever wrapper around agent-bench's async Retriever."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_bench.langchain_baseline.retriever import AgentBenchRetriever


def _make_mock_retriever(results=None):
    """Create a mock of agent_bench.rag.retriever.Retriever."""
    retriever = MagicMock()
    if results is None:
        # Default: one result with known fields
        result = MagicMock()
        result.chunk.content = "Path parameters use curly braces."
        result.chunk.source = "fastapi_path_params.md"
        result.chunk.id = "chunk_001"
        result.score = 0.85
        result.rank = 1
        results = [result]
    retriever.search = AsyncMock(return_value=results)
    return retriever


async def test_returns_langchain_documents():
    mock_ret = _make_mock_retriever()
    wrapper = AgentBenchRetriever(retriever=mock_ret, top_k=5)
    docs = await wrapper.ainvoke("path parameters")

    assert len(docs) == 1
    assert docs[0].page_content == "Path parameters use curly braces."
    assert docs[0].metadata["source"] == "fastapi_path_params.md"
    assert docs[0].metadata["chunk_id"] == "chunk_001"
    assert docs[0].metadata["score"] == 0.85


async def test_passes_top_k_to_underlying_retriever():
    mock_ret = _make_mock_retriever()
    wrapper = AgentBenchRetriever(retriever=mock_ret, top_k=3)
    await wrapper.ainvoke("test")
    mock_ret.search.assert_called_once_with("test", top_k=3)


async def test_handles_empty_results():
    mock_ret = _make_mock_retriever(results=[])
    wrapper = AgentBenchRetriever(retriever=mock_ret, top_k=5)
    docs = await wrapper.ainvoke("nonsense")
    assert docs == []


async def test_multiple_results_preserve_order():
    r1 = MagicMock()
    r1.chunk.content = "First"
    r1.chunk.source = "a.md"
    r1.chunk.id = "c1"
    r1.score = 0.9

    r2 = MagicMock()
    r2.chunk.content = "Second"
    r2.chunk.source = "b.md"
    r2.chunk.id = "c2"
    r2.score = 0.7

    mock_ret = _make_mock_retriever(results=[r1, r2])
    wrapper = AgentBenchRetriever(retriever=mock_ret, top_k=5)
    docs = await wrapper.ainvoke("test")

    assert len(docs) == 2
    assert docs[0].page_content == "First"
    assert docs[1].page_content == "Second"
```

**Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_langchain_baseline/test_retriever.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'agent_bench.langchain_baseline.retriever'`

**Step 4: Implement the retriever wrapper**

Create `agent_bench/langchain_baseline/retriever.py`:

```python
"""LangChain BaseRetriever wrapping agent-bench's async hybrid retriever."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, List

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever

if TYPE_CHECKING:
    from agent_bench.rag.retriever import Retriever


class AgentBenchRetriever(BaseRetriever):
    """Wraps agent-bench's async Retriever as a LangChain retriever.

    Delegates to Retriever.search() which returns list[SearchResult].
    Each SearchResult has .chunk.content, .chunk.source, .chunk.id, .score.
    """

    retriever: Any  # agent_bench.rag.retriever.Retriever (Pydantic can't validate it)
    top_k: int = 5

    model_config = {"arbitrary_types_allowed": True}

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[LCDocument]:
        results = await self.retriever.search(query, top_k=self.top_k)
        return [
            LCDocument(
                page_content=r.chunk.content,
                metadata={
                    "source": r.chunk.source,
                    "chunk_id": r.chunk.id,
                    "score": r.score,
                },
            )
            for r in results
        ]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[LCDocument]:
        """Sync fallback: runs async implementation in a new event loop thread."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._aget_relevant_documents(
                    query,
                    run_manager=AsyncCallbackManagerForRetrieverRun.get_noop_manager(),
                )
            )
        finally:
            loop.close()
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_langchain_baseline/test_retriever.py -v`

Expected: 4 passed

**Step 6: Commit**

```bash
git add agent_bench/langchain_baseline/__init__.py agent_bench/langchain_baseline/retriever.py tests/test_langchain_baseline/__init__.py tests/test_langchain_baseline/test_retriever.py
git commit -m "feat: langchain retriever wrapper over existing async hybrid retriever"
```

---

## Task 3: Search Tool with Metadata Capture

**Files:**
- Create: `agent_bench/langchain_baseline/tools.py`
- Create: `tests/test_langchain_baseline/test_tools.py`

The search tool needs to capture retrieval metadata (ranked sources, source chunks) in a side channel so the evaluation runner can compute P@5, R@5, and citation accuracy without parsing strings. This is done via a stateful `LangChainSearchTool` class.

**Step 1: Write the failing test**

Create `tests/test_langchain_baseline/test_tools.py`:

```python
"""Tests for LangChain tool wrappers."""

from unittest.mock import AsyncMock, MagicMock

from langchain_core.documents import Document as LCDocument

from agent_bench.langchain_baseline.tools import LangChainSearchTool, create_calculator_tool


# --- Search tool ---


def _make_mock_lc_retriever(docs=None):
    """Mock an AgentBenchRetriever (LangChain retriever)."""
    ret = MagicMock()
    if docs is None:
        docs = [
            LCDocument(
                page_content="Path params use curly braces.",
                metadata={"source": "fastapi_path_params.md", "chunk_id": "c1", "score": 0.9},
            ),
            LCDocument(
                page_content="Query params are parsed from URL.",
                metadata={"source": "fastapi_query_params.md", "chunk_id": "c2", "score": 0.7},
            ),
        ]
    ret.ainvoke = AsyncMock(return_value=docs)
    return ret


async def test_search_tool_returns_formatted_passages():
    mock_ret = _make_mock_lc_retriever()
    search = LangChainSearchTool(mock_ret)
    tool = search.as_tool()

    result = await tool.ainvoke({"query": "path parameters"})

    assert "[1] (fastapi_path_params.md):" in result
    assert "[2] (fastapi_query_params.md):" in result
    assert "curly braces" in result


async def test_search_tool_captures_ranked_sources():
    mock_ret = _make_mock_lc_retriever()
    search = LangChainSearchTool(mock_ret)
    tool = search.as_tool()

    await tool.ainvoke({"query": "test"})

    assert search.last_ranked_sources == [
        "fastapi_path_params.md",
        "fastapi_query_params.md",
    ]


async def test_search_tool_captures_source_chunks():
    mock_ret = _make_mock_lc_retriever()
    search = LangChainSearchTool(mock_ret)
    tool = search.as_tool()

    await tool.ainvoke({"query": "test"})

    assert search.last_source_chunks == [
        "Path params use curly braces.",
        "Query params are parsed from URL.",
    ]


async def test_search_tool_deduplicates_sources():
    docs = [
        LCDocument(page_content="A", metadata={"source": "x.md", "chunk_id": "c1", "score": 0.9}),
        LCDocument(page_content="B", metadata={"source": "x.md", "chunk_id": "c2", "score": 0.8}),
    ]
    mock_ret = _make_mock_lc_retriever(docs)
    search = LangChainSearchTool(mock_ret)
    tool = search.as_tool()

    await tool.ainvoke({"query": "test"})

    assert search.last_sources == ["x.md"]
    assert search.last_ranked_sources == ["x.md", "x.md"]


async def test_search_tool_handles_no_results():
    mock_ret = _make_mock_lc_retriever(docs=[])
    search = LangChainSearchTool(mock_ret)
    tool = search.as_tool()

    result = await tool.ainvoke({"query": "nothing"})
    assert "No relevant documents found" in result
    assert search.last_ranked_sources == []


async def test_search_tool_accumulates_across_multiple_calls():
    """If the agent calls search twice in one turn, metadata accumulates."""
    docs1 = [
        LCDocument(page_content="A", metadata={"source": "a.md", "chunk_id": "c1", "score": 0.9}),
    ]
    docs2 = [
        LCDocument(page_content="B", metadata={"source": "b.md", "chunk_id": "c2", "score": 0.8}),
    ]
    mock_ret = MagicMock()
    mock_ret.ainvoke = AsyncMock(side_effect=[docs1, docs2])

    search = LangChainSearchTool(mock_ret)
    tool = search.as_tool()

    await tool.ainvoke({"query": "first"})
    await tool.ainvoke({"query": "second"})

    assert search.last_ranked_sources == ["a.md", "b.md"]
    assert search.last_source_chunks == ["A", "B"]
    assert search.last_sources == ["a.md", "b.md"]


async def test_search_tool_reset_clears_state():
    mock_ret = _make_mock_lc_retriever()
    search = LangChainSearchTool(mock_ret)
    tool = search.as_tool()

    await tool.ainvoke({"query": "test"})
    assert len(search.last_ranked_sources) > 0

    search.reset()
    assert search.last_ranked_sources == []
    assert search.last_source_chunks == []
    assert search.last_sources == []


# --- Calculator tool ---


async def test_calculator_evaluates_expression():
    tool = create_calculator_tool()
    result = await tool.ainvoke({"expression": "2 + 3 * 4"})
    assert "14" in result


async def test_calculator_handles_invalid_expression():
    tool = create_calculator_tool()
    result = await tool.ainvoke({"expression": "not_a_number"})
    assert "Error" in result or "error" in result
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_langchain_baseline/test_tools.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement the tools module**

Create `agent_bench/langchain_baseline/tools.py`:

```python
"""LangChain tool wrappers with metadata capture for evaluation metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from simpleeval import simple_eval

if TYPE_CHECKING:
    from agent_bench.langchain_baseline.retriever import AgentBenchRetriever


# --- Search tool with metadata side-channel ---


class SearchInput(BaseModel):
    query: str = Field(description="The search query to find relevant documentation")


class LangChainSearchTool:
    """Stateful search tool that captures retrieval metadata for evaluation.

    After each invocation, `last_ranked_sources`, `last_source_chunks`,
    and `last_sources` contain the retrieval data needed to compute
    P@5, R@5, and citation accuracy using the existing metric functions.
    Call `reset()` before each new question.
    """

    def __init__(self, retriever: AgentBenchRetriever) -> None:
        self._retriever = retriever
        self.last_ranked_sources: list[str] = []
        self.last_source_chunks: list[str] = []
        self.last_sources: list[str] = []

    def reset(self) -> None:
        self.last_ranked_sources = []
        self.last_source_chunks = []
        self.last_sources = []

    async def _search_async(self, query: str) -> str:
        docs = await self._retriever.ainvoke(query)

        # Accumulate across multiple tool calls within one question.
        # The runner calls reset() between questions.

        if not docs:
            return "No relevant documents found."

        lines = []
        for i, d in enumerate(docs, 1):
            src = d.metadata["source"]
            self.last_ranked_sources.append(src)
            self.last_source_chunks.append(d.page_content)
            if src not in self.last_sources:
                self.last_sources.append(src)
            lines.append(f"[{i}] ({src}): {d.page_content}")

        return "\n\n".join(lines)

    def _search_sync(self, query: str) -> str:
        """Sync fallback — runs async search in a new event loop."""
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._search_async(query))
        finally:
            loop.close()

    def as_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self._search_sync,
            coroutine=self._search_async,
            name="search_documents",
            description=(
                "Search the technical documentation corpus for relevant passages. "
                "Returns the most relevant document chunks with source attribution."
            ),
            args_schema=SearchInput,
        )


# --- Calculator tool ---


class CalcInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate, e.g. '2 + 3 * 4'")


def create_calculator_tool() -> StructuredTool:
    def calculate(expression: str) -> str:
        try:
            result = simple_eval(expression)
            return str(result)
        except Exception as e:
            return f"Error evaluating '{expression}': {e}"

    return StructuredTool.from_function(
        func=calculate,
        name="calculator",
        description="Evaluate mathematical expressions. Use for any numerical computations.",
        args_schema=CalcInput,
    )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_langchain_baseline/test_tools.py -v`

Expected: 10 passed

**Step 5: Commit**

```bash
git add agent_bench/langchain_baseline/tools.py tests/test_langchain_baseline/test_tools.py
git commit -m "feat: langchain search tool with metadata capture + calculator"
```

---

## Task 4: Agent Factory

**Files:**
- Create: `agent_bench/langchain_baseline/agent.py`
- Create: `tests/test_langchain_baseline/test_agent.py`

**Step 1: Write the failing test**

Create `tests/test_langchain_baseline/test_agent.py`:

```python
"""Tests for LangChain agent factory."""

from unittest.mock import MagicMock, patch

from langchain.agents import AgentExecutor
from langchain_core.tools import StructuredTool

from agent_bench.langchain_baseline.agent import create_langchain_agent


def _make_dummy_tool():
    return StructuredTool.from_function(
        func=lambda query: "result",
        name="test_tool",
        description="A test tool",
    )


@patch("agent_bench.langchain_baseline.agent.ChatOpenAI")
def test_creates_agent_executor_openai(mock_chat):
    mock_chat.return_value = MagicMock()
    tool = _make_dummy_tool()

    executor = create_langchain_agent(
        tools=[tool],
        provider="openai",
    )

    assert isinstance(executor, AgentExecutor)
    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs.kwargs["temperature"] == 0.0


@patch("agent_bench.langchain_baseline.agent.ChatAnthropic")
def test_creates_agent_executor_anthropic(mock_chat):
    mock_chat.return_value = MagicMock()
    tool = _make_dummy_tool()

    executor = create_langchain_agent(
        tools=[tool],
        provider="anthropic",
    )

    assert isinstance(executor, AgentExecutor)
    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args
    assert call_kwargs.kwargs["model"] == "claude-haiku-4-5-20251001"


@patch("agent_bench.langchain_baseline.agent.ChatOpenAI")
def test_custom_model_override(mock_chat):
    mock_chat.return_value = MagicMock()
    tool = _make_dummy_tool()

    create_langchain_agent(
        tools=[tool],
        provider="openai",
        model="gpt-4o",
    )

    call_kwargs = mock_chat.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4o"


def test_unknown_provider_raises():
    import pytest

    tool = _make_dummy_tool()
    with pytest.raises(ValueError, match="Unknown provider"):
        create_langchain_agent(tools=[tool], provider="unknown")


@patch("agent_bench.langchain_baseline.agent.ChatOpenAI")
def test_uses_custom_system_prompt(mock_chat):
    mock_chat.return_value = MagicMock()
    tool = _make_dummy_tool()

    executor = create_langchain_agent(
        tools=[tool],
        provider="openai",
        system_prompt="Custom prompt here",
    )

    assert isinstance(executor, AgentExecutor)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_langchain_baseline/test_agent.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement the agent factory**

Create `agent_bench/langchain_baseline/agent.py`:

```python
"""LangChain tool-calling agent factory.

Uses native function calling (not ReAct text parsing) for a fair
apples-to-apples comparison with the custom pipeline.
"""

from __future__ import annotations

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

_DEFAULT_SYSTEM_PROMPT = (
    "You are a technical documentation assistant. You have access to tools "
    "that let you search a documentation corpus and perform calculations.\n\n"
    "Rules:\n"
    "- Use search_documents to find relevant information before answering.\n"
    "- Base your answer ONLY on the retrieved documents.\n"
    "- Cite sources inline as [source: filename.md] for each claim.\n"
    "- If the documents don't contain the answer, respond with: "
    '"The documentation does not contain information about this topic."\n'
    "- Use calculator for any numerical computations.\n"
    "- Be concise and precise."
)


def create_langchain_agent(
    tools: list[BaseTool],
    provider: str = "openai",
    model: str | None = None,
    temperature: float = 0.0,
    system_prompt: str | None = None,
    max_iterations: int = 5,
) -> AgentExecutor:
    """Create a LangChain tool-calling agent.

    Args:
        tools: LangChain tools for the agent.
        provider: "openai" or "anthropic".
        model: Model name override. Defaults to gpt-4o-mini / claude-haiku-4-5-20251001.
        temperature: LLM temperature (0.0 for reproducibility).
        system_prompt: System prompt. Defaults to the tech_docs task prompt.
        max_iterations: Max tool-use iterations before forcing a final answer.
    """
    if provider == "openai":
        llm = ChatOpenAI(model=model or "gpt-4o-mini", temperature=temperature)
    elif provider == "anthropic":
        llm = ChatAnthropic(
            model=model or "claude-haiku-4-5-20251001", temperature=temperature
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt or _DEFAULT_SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_langchain_baseline/test_agent.py -v`

Expected: 5 passed

**Step 5: Commit**

```bash
git add agent_bench/langchain_baseline/agent.py tests/test_langchain_baseline/test_agent.py
git commit -m "feat: langchain tool-calling agent factory"
```

---

## Task 5: Evaluation Runner

**Files:**
- Create: `agent_bench/langchain_baseline/runner.py`
- Create: `tests/test_langchain_baseline/test_runner.py`

This runner produces `EvalResult` objects using the same metric functions as the existing harness, enabling direct use of `generate_report()`.

**Step 1: Write the failing test**

Create `tests/test_langchain_baseline/test_runner.py`:

```python
"""Tests for LangChain evaluation runner."""

from unittest.mock import AsyncMock, MagicMock

from agent_bench.langchain_baseline.runner import (
    extract_tools_used,
    run_langchain_evaluation,
)
from agent_bench.langchain_baseline.tools import LangChainSearchTool


# --- Unit tests for helper functions ---


def test_extract_tools_used_from_intermediate_steps():
    step1_action = MagicMock()
    step1_action.tool = "search_documents"
    step2_action = MagicMock()
    step2_action.tool = "calculator"

    steps = [(step1_action, "result1"), (step2_action, "result2")]
    assert extract_tools_used(steps) == ["search_documents", "calculator"]


def test_extract_tools_used_empty_steps():
    assert extract_tools_used([]) == []


# --- Integration test with mock agent executor ---


async def test_runner_produces_eval_results():
    # Mock agent executor
    agent_executor = MagicMock()
    agent_executor.ainvoke = AsyncMock(return_value={
        "output": "Path params use curly braces. [source: fastapi_path_params.md]",
        "intermediate_steps": [
            (MagicMock(tool="search_documents"), "tool output"),
        ],
    })

    # Mock search tool state
    mock_lc_retriever = MagicMock()
    search_tool = LangChainSearchTool(mock_lc_retriever)
    search_tool.last_ranked_sources = ["fastapi_path_params.md"]
    search_tool.last_source_chunks = ["Path params use curly braces."]
    search_tool.last_sources = ["fastapi_path_params.md"]

    golden_path = "agent_bench/evaluation/datasets/tech_docs_golden.json"

    results = await run_langchain_evaluation(
        agent_executor=agent_executor,
        search_tool_state=search_tool,
        golden_path=golden_path,
        provider_name="openai",
        max_questions=2,  # only run first 2 for speed
    )

    assert len(results) == 2
    r = results[0]
    assert r.question_id == "q001"
    assert r.question == "How do you define a path parameter in FastAPI?"
    assert r.category == "retrieval"
    assert r.answer != ""
    assert r.retrieval_precision >= 0.0
    assert r.retrieval_recall >= 0.0


async def test_runner_handles_agent_error():
    agent_executor = MagicMock()
    agent_executor.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))

    mock_lc_retriever = MagicMock()
    search_tool = LangChainSearchTool(mock_lc_retriever)

    golden_path = "agent_bench/evaluation/datasets/tech_docs_golden.json"

    results = await run_langchain_evaluation(
        agent_executor=agent_executor,
        search_tool_state=search_tool,
        golden_path=golden_path,
        provider_name="openai",
        max_questions=1,
    )

    assert len(results) == 1
    assert "ERROR" in results[0].answer
    assert results[0].tool_calls_made == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_langchain_baseline/test_runner.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement the runner**

Create `agent_bench/langchain_baseline/runner.py`:

```python
"""Evaluation runner: LangChain agent -> EvalResult (same format as existing harness)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from agent_bench.core.types import TokenUsage
from agent_bench.evaluation.harness import EvalResult, load_golden_dataset
from agent_bench.evaluation.metrics import (
    citation_accuracy,
    grounded_refusal,
    keyword_hit_rate,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
)

if TYPE_CHECKING:
    from langchain.agents import AgentExecutor

    from agent_bench.langchain_baseline.tools import LangChainSearchTool


def extract_tools_used(intermediate_steps: list) -> list[str]:
    """Extract tool names from LangChain intermediate steps.

    Each step is a (AgentAction, observation) tuple.
    """
    return [step[0].tool for step in intermediate_steps if hasattr(step[0], "tool")]


async def run_langchain_evaluation(
    agent_executor: AgentExecutor,
    search_tool_state: LangChainSearchTool,
    golden_path: str | Path,
    provider_name: str,
    max_questions: int | None = None,
) -> list[EvalResult]:
    """Run golden dataset through LangChain agent, producing EvalResult objects.

    Uses the same metric functions as agent_bench.evaluation.harness, so results
    are directly comparable and can be fed into generate_report().

    Args:
        agent_executor: Configured LangChain AgentExecutor.
        search_tool_state: The LangChainSearchTool instance (for metadata capture).
        golden_path: Path to the golden dataset JSON.
        provider_name: Provider name for reporting (e.g. "openai").
        max_questions: Limit number of questions (for testing). None = all.
    """
    questions = load_golden_dataset(golden_path)
    if max_questions is not None:
        questions = questions[:max_questions]

    results: list[EvalResult] = []

    for q in questions:
        search_tool_state.reset()
        start = time.perf_counter()

        try:
            response = await agent_executor.ainvoke({"input": q.question})
            latency_ms = (time.perf_counter() - start) * 1000

            answer = response.get("output", "")
            steps = response.get("intermediate_steps", [])
            tools_used = extract_tools_used(steps)

            ranked_sources = list(search_tool_state.last_ranked_sources)
            deduped_sources = list(search_tool_state.last_sources)

            result = EvalResult(
                question_id=q.id,
                question=q.question,
                category=q.category,
                difficulty=q.difficulty,
                retrieval_precision=retrieval_precision_at_k(
                    ranked_sources, q.expected_sources
                ),
                retrieval_recall=retrieval_recall_at_k(
                    ranked_sources, q.expected_sources
                ),
                keyword_hit_rate=keyword_hit_rate(answer, q.expected_answer_keywords),
                has_source_citation=len(deduped_sources) > 0,
                grounded_refusal=grounded_refusal(
                    answer, q.category, deduped_sources
                ),
                citation_accuracy=citation_accuracy(answer, deduped_sources),
                calculator_used_correctly=(
                    ("calculator" in tools_used) if q.requires_calculator else True
                ),
                tool_calls_made=len(tools_used),
                latency_ms=latency_ms,
                tokens_used=TokenUsage(
                    input_tokens=0, output_tokens=0, estimated_cost_usd=0.0
                ),
                answer=answer,
                retrieved_sources=ranked_sources,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            result = EvalResult(
                question_id=q.id,
                question=q.question,
                category=q.category,
                difficulty=q.difficulty,
                retrieval_precision=0.0,
                retrieval_recall=0.0,
                keyword_hit_rate=0.0,
                has_source_citation=False,
                grounded_refusal=q.category != "out_of_scope",
                citation_accuracy=1.0,
                calculator_used_correctly=not q.requires_calculator,
                tool_calls_made=0,
                latency_ms=latency_ms,
                tokens_used=TokenUsage(
                    input_tokens=0, output_tokens=0, estimated_cost_usd=0.0
                ),
                answer=f"ERROR: {e}",
                retrieved_sources=[],
            )

        results.append(result)

    return results
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_langchain_baseline/test_runner.py -v`

Expected: 4 passed

**Step 5: Commit**

```bash
git add agent_bench/langchain_baseline/runner.py tests/test_langchain_baseline/test_runner.py
git commit -m "feat: langchain evaluation runner producing EvalResult objects"
```

---

## Task 6: CLI Script and Makefile Target

**Files:**
- Create: `scripts/run_langchain_eval.py`
- Modify: `Makefile:1-32`

**Step 1: Create the CLI script**

Create `scripts/run_langchain_eval.py`:

```python
"""Run LangChain baseline evaluation against the golden dataset.

Usage:
    python scripts/run_langchain_eval.py --provider openai
    python scripts/run_langchain_eval.py --provider anthropic
    python scripts/run_langchain_eval.py --provider openai --max-questions 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_bench.core.config import load_config, load_task_config
from agent_bench.evaluation.report import generate_report, save_report
from agent_bench.langchain_baseline.agent import create_langchain_agent
from agent_bench.langchain_baseline.retriever import AgentBenchRetriever
from agent_bench.langchain_baseline.runner import run_langchain_evaluation
from agent_bench.langchain_baseline.tools import LangChainSearchTool, create_calculator_tool
from agent_bench.rag.embedder import Embedder
from agent_bench.rag.retriever import Retriever
from agent_bench.rag.store import HybridStore


async def main_async(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config) if args.config else None)
    task = load_task_config("tech_docs")

    # Build existing RAG pipeline (same as scripts/evaluate.py)
    store = HybridStore.load(config.rag.store_path, rrf_k=config.rag.retrieval.rrf_k)
    embedder = Embedder(model_name=config.embedding.model, cache_dir=config.embedding.cache_dir)

    reranker = None
    if config.rag.reranker.enabled:
        from agent_bench.rag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=config.rag.reranker.model_name)

    retriever = Retriever(
        embedder=embedder,
        store=store,
        default_strategy=config.rag.retrieval.strategy,
        candidates_per_system=config.rag.retrieval.candidates_per_system,
        reranker=reranker,
        reranker_top_k=config.rag.reranker.top_k,
    )

    # Wrap in LangChain components
    lc_retriever = AgentBenchRetriever(retriever=retriever, top_k=config.rag.retrieval.top_k)
    search_tool = LangChainSearchTool(lc_retriever)
    calc_tool = create_calculator_tool()

    agent_executor = create_langchain_agent(
        tools=[search_tool.as_tool(), calc_tool],
        provider=args.provider,
        system_prompt=task.system_prompt,
    )

    # Run evaluation
    golden_path = config.evaluation.golden_dataset
    print(f"Running LangChain baseline evaluation...")
    print(f"  Provider:  {args.provider}")
    print(f"  Store:     {store.stats().total_chunks} chunks")
    print(f"  Golden:    {golden_path}")
    if args.max_questions:
        print(f"  Limit:     {args.max_questions} questions")
    print()

    results = await run_langchain_evaluation(
        agent_executor=agent_executor,
        search_tool_state=search_tool,
        golden_path=golden_path,
        provider_name=args.provider,
        max_questions=args.max_questions,
    )

    # Save raw results JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_data = [r.model_dump() for r in results]
    output_path.write_text(json.dumps(results_data, indent=2, default=str))
    print(f"Results JSON: {output_path}")

    # Generate markdown report (reuses existing report generator)
    report = generate_report(
        results,
        provider_name=f"langchain-{args.provider}",
        corpus_size=store.stats().unique_sources,
    )
    report_path = Path(f"docs/langchain_benchmark_{args.provider}.md")
    save_report(report, report_path)
    print(f"Report:      {report_path}")

    # Print summary
    positive = [r for r in results if r.category != "out_of_scope"]
    errors = [r for r in results if r.answer.startswith("ERROR")]
    avg_p5 = sum(r.retrieval_precision for r in positive) / max(len(positive), 1)
    avg_r5 = sum(r.retrieval_recall for r in positive) / max(len(positive), 1)
    avg_khr = sum(r.keyword_hit_rate for r in positive) / max(len(positive), 1)
    avg_lat = sum(r.latency_ms for r in results) / max(len(results), 1)

    print(f"\nSummary ({len(results)} questions, {len(errors)} errors):")
    print(f"  Avg P@5:     {avg_p5:.2f}")
    print(f"  Avg R@5:     {avg_r5:.2f}")
    print(f"  Avg KHR:     {avg_khr:.2f}")
    print(f"  Avg latency: {avg_lat:,.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangChain baseline evaluation")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
    )
    parser.add_argument("--config", default=None, help="Config YAML path")
    parser.add_argument("--output", default=".cache/langchain_eval_results.json")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions (for testing)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
```

**Step 2: Add Makefile target**

Add after the existing `benchmark` target in `Makefile`:

```makefile
evaluate-langchain:
	$(PYTHON) scripts/run_langchain_eval.py --provider openai
```

**Step 3: Run script with --help to verify it loads**

Run: `python scripts/run_langchain_eval.py --help`

Expected: Shows argparse help text without import errors.

**Step 4: Commit**

```bash
git add scripts/run_langchain_eval.py Makefile
git commit -m "feat: langchain evaluation CLI script and Makefile target"
```

---

## Task 7: Verify No Regressions

**Step 1: Run the full existing test suite**

Run: `python -m pytest tests/ -v --tb=short`

Expected: All existing tests pass (145+). New tests also pass. Zero failures.

**Step 2: Run linter**

Run: `ruff check agent_bench/langchain_baseline/ tests/test_langchain_baseline/`

If any lint issues, fix them.

**Step 3: Commit any lint fixes**

```bash
git add -A
git commit -m "fix: lint issues in langchain baseline"
```

---

## Task 8: Run Evaluation and Populate Comparison Table

**This task requires API keys and the ingested store at `.cache/store`.**

**Step 1: Run with OpenAI (quick test first)**

Run: `python scripts/run_langchain_eval.py --provider openai --max-questions 3`

Verify: Script completes, prints summary with real numbers, produces JSON output.

**Step 2: Run full OpenAI evaluation**

Run: `python scripts/run_langchain_eval.py --provider openai`

Expected: 27 questions evaluated, report at `docs/langchain_benchmark_openai.md`.

**Step 3: (Optional) Run with Anthropic**

Run: `python scripts/run_langchain_eval.py --provider anthropic`

**Step 4: Create comparison table**

Create `results/comparison_custom_vs_langchain.md` with the real numbers from both the existing benchmark report (`docs/benchmark_report.md`) and the new LangChain report(s).

**Step 5: Commit**

```bash
git add docs/langchain_benchmark_*.md results/comparison_custom_vs_langchain.md
git commit -m "feat: langchain baseline evaluation results"
```

---

## Task 9: Update README

**Files:**
- Modify: `README.md`

**Step 1: Add comparison section**

Add a new `## Framework Comparison: Custom vs. LangChain` section to `README.md` after the existing evaluation section. Include:

- One-paragraph explanation of the comparison approach
- The comparison results table from `results/comparison_custom_vs_langchain.md`
- 2-3 key takeaways (fill in after seeing real results)

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add langchain baseline comparison to README"
```

---

## Reference: Key Interfaces

These are the existing interfaces the plan builds against. Consult these if anything is unclear during implementation.

**`Retriever.search()`** — `agent_bench/rag/retriever.py:33-77`
```python
async def search(self, query: str, top_k: int = 5, strategy: str | None = None) -> list[SearchResult]
```

**`SearchResult`** — `agent_bench/rag/store.py:19-25`
```python
class SearchResult(BaseModel):
    chunk: Chunk       # .content, .source, .id
    score: float
    rank: int
    retrieval_strategy: str
```

**`Chunk`** — `agent_bench/rag/chunker.py:11-16`
```python
class Chunk(BaseModel):
    id: str
    content: str
    source: str        # bare filename, e.g. "fastapi_path_params.md"
    chunk_index: int
    metadata: dict
```

**`EvalResult`** — `agent_bench/evaluation/harness.py:36-57`
```python
class EvalResult(BaseModel):
    question_id: str
    question: str
    category: str
    difficulty: str
    retrieval_precision: float
    retrieval_recall: float
    keyword_hit_rate: float
    has_source_citation: bool
    grounded_refusal: bool
    citation_accuracy: float
    calculator_used_correctly: bool
    tool_calls_made: int
    latency_ms: float
    tokens_used: TokenUsage
    answer: str = ""
    retrieved_sources: list[str] = []
    faithfulness: float | None = None
    correctness: float | None = None
```

**Golden dataset** — `agent_bench/evaluation/datasets/tech_docs_golden.json`
- 27 questions: 19 retrieval, 3 calculation, 5 out_of_scope
- `expected_sources` are bare filenames (e.g. `"fastapi_path_params.md"`)

**System prompt** — `configs/tasks/tech_docs.yaml`
- References tools by name: `search_documents`, `calculator`
- Citation format: `[source: filename.md]`

**Models (match existing pipeline for fair comparison):**
- OpenAI: `gpt-4o-mini`
- Anthropic: `claude-haiku-4-5-20251001`
