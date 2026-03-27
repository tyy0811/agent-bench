"""LangChain BaseRetriever wrapping agent-bench's async hybrid retriever."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, List

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever


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
        """Sync fallback — safe even when called from inside a running event loop."""
        coro = self._aget_relevant_documents(
            query,
            run_manager=AsyncCallbackManagerForRetrieverRun.get_noop_manager(),
        )
        # If there's already a running loop (e.g. inside asyncio.run), we can't
        # call loop.run_until_complete in the same thread.  Spin up a dedicated
        # thread with its own loop to avoid RuntimeError.
        result: list[LCDocument] = []
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
