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
