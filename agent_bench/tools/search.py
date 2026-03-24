"""Search tool: RAG retrieval over the document corpus."""

from __future__ import annotations

from typing import Protocol

from agent_bench.tools.base import Tool, ToolOutput


class SearchResult(Protocol):
    """Protocol for retriever search results (defined fully in rag.store)."""

    @property
    def chunk(self) -> object: ...

    @property
    def score(self) -> float: ...


class Retriever(Protocol):
    """Protocol for the retriever dependency (defined fully in rag.retriever)."""

    async def search(self, query: str, top_k: int = 5, strategy: str | None = None) -> list: ...


class SearchTool(Tool):
    """Search the document corpus and return relevant passages."""

    name = "search_documents"
    description = (
        "Search the technical documentation corpus for relevant passages. "
        "Returns the most relevant document chunks with source attribution."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant documentation",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 5)",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        retriever: Retriever,
        default_top_k: int = 5,
        default_strategy: str = "hybrid",
    ) -> None:
        self._retriever = retriever
        self.default_top_k = default_top_k
        self.default_strategy = default_strategy

    async def execute(self, **kwargs: object) -> ToolOutput:
        query = str(kwargs.get("query", ""))
        top_k_val = kwargs.get("top_k", self.default_top_k)
        try:
            top_k: int = top_k_val if isinstance(top_k_val, int) else int(str(top_k_val))
        except (ValueError, TypeError):
            top_k = self.default_top_k
        # _strategy is injected by the orchestrator per-request (not from LLM args)
        strategy = str(kwargs.get("_strategy", self.default_strategy))

        if not query:
            return ToolOutput(success=False, result="No query provided")

        results = await self._retriever.search(query, top_k=top_k, strategy=strategy)

        if not results:
            return ToolOutput(
                success=True,
                result="No relevant documents found.",
                metadata={"sources": []},
            )

        # Format as numbered passages with filename attribution
        lines = []
        sources = []
        for i, r in enumerate(results, 1):
            source = r.chunk.source
            content = r.chunk.content
            lines.append(f"[{i}] ({source}): {content}")
            if source not in sources:
                sources.append(source)

        return ToolOutput(
            success=True,
            result="\n\n".join(lines),
            metadata={"sources": sources},
        )
