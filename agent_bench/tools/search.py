"""Search tool: RAG retrieval over the document corpus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import structlog

from agent_bench.rag.retriever import RetrievalResult
from agent_bench.tools.base import Tool, ToolOutput

if TYPE_CHECKING:
    from agent_bench.security.pii_redactor import PIIRedactor

log = structlog.get_logger()


class SearchResult(Protocol):
    """Protocol for retriever search results (defined fully in rag.store)."""

    @property
    def chunk(self) -> object: ...

    @property
    def score(self) -> float: ...


class Retriever(Protocol):
    """Protocol for the retriever dependency (defined fully in rag.retriever)."""

    async def search(self, query: str, top_k: int = 5, strategy: str | None = None) -> RetrievalResult: ...


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
        refusal_threshold: float = 0.0,
        pii_redactor: PIIRedactor | None = None,
    ) -> None:
        self._retriever = retriever
        self.default_top_k = default_top_k
        self.default_strategy = default_strategy
        self.refusal_threshold = refusal_threshold
        self._pii_redactor = pii_redactor

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

        retrieval_result = await self._retriever.search(query, top_k=top_k, strategy=strategy)
        results = retrieval_result.results
        pre_rerank_count = retrieval_result.pre_rerank_count

        if not results:
            return ToolOutput(
                success=True,
                result="No relevant documents found.",
                metadata={"sources": [], "pre_rerank_count": pre_rerank_count,
                          "chunks": [], "pii_redactions_count": 0},
            )

        # Compute max retrieval score for refusal gate
        max_score = max(r.score for r in results)
        log.info("retrieval_scores", query=query, max_score=max_score, num_results=len(results))

        # Refusal gate: if max score is below threshold, refuse to answer
        if self.refusal_threshold > 0 and max_score < self.refusal_threshold:
            log.info("retrieval_refused", query=query, max_score=max_score,
                     threshold=self.refusal_threshold)
            top = results[0]
            return ToolOutput(
                success=True,
                result="No relevant documents found for this query.",
                metadata={
                    "sources": [], "max_score": max_score, "refused": True,
                    "refusal_threshold": self.refusal_threshold,
                    "pre_rerank_count": pre_rerank_count,
                    "chunks": [{"source": top.chunk.source,
                                "score": rs if (rs := getattr(top, 'rerank_score', None)) is not None else top.score,
                                "preview": top.chunk.content[:120]}],
                    "pii_redactions_count": 0,
                },
            )

        # Format as numbered passages with filename attribution
        lines = []
        sources = []
        ranked_sources = []  # preserves rank order with duplicates
        source_chunks = []  # raw chunk text for LLM judge
        chunk_details = []
        total_pii_redactions = 0
        for i, r in enumerate(results, 1):
            source = r.chunk.source
            content = r.chunk.content
            # PII redaction: scrub retrieved chunks before they enter the LLM prompt
            if self._pii_redactor is not None:
                redacted = self._pii_redactor.redact(content)
                total_pii_redactions += redacted.redactions_count
                content = redacted.text
            lines.append(f"[{i}] ({source}): {content}")
            ranked_sources.append(source)
            source_chunks.append(content)
            chunk_details.append({
                "source": source,
                "score": rs if (rs := getattr(r, 'rerank_score', None)) is not None else r.score,
                "preview": content[:120],
            })
            if source not in sources:
                sources.append(source)

        return ToolOutput(
            success=True,
            result="\n\n".join(lines),
            metadata={
                "sources": sources,
                "ranked_sources": ranked_sources,
                "source_chunks": source_chunks,
                "max_score": max_score,
                "pre_rerank_count": pre_rerank_count,
                "chunks": chunk_details,
                "pii_redactions_count": total_pii_redactions,
            },
        )
