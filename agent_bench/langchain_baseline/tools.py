"""LangChain tool wrappers with metadata capture for evaluation metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        """Sync fallback — safe even when called from inside a running event loop."""
        import asyncio
        import threading

        coro = self._search_async(query)
        result: str = ""
        exc: BaseException | None = None

        def _run() -> None:
            nonlocal result, exc
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(coro)
            except BaseException as e:
                exc = e
            finally:
                loop.close()

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()
        if exc is not None:
            raise exc
        return result

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
    expression: str = Field(
        description="Mathematical expression to evaluate, e.g. '2 + 3 * 4'"
    )


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
