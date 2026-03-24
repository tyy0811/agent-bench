"""Retrieval orchestration: embed query → search store → return results."""

from __future__ import annotations

from typing import Literal, cast

from agent_bench.rag.embedder import Embedder
from agent_bench.rag.store import HybridStore, SearchResult


class Retriever:
    """Thin glue between embedder and store."""

    def __init__(
        self,
        embedder: Embedder,
        store: HybridStore,
        default_strategy: Literal["semantic", "keyword", "hybrid"] = "hybrid",
        candidates_per_system: int = 10,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._default_strategy = default_strategy
        self._candidates_per_system = candidates_per_system

    async def search(
        self,
        query: str,
        top_k: int = 5,
        strategy: str | None = None,
    ) -> list[SearchResult]:
        """Embed query and search the store."""
        strat: Literal["semantic", "keyword", "hybrid"] = cast(
            Literal["semantic", "keyword", "hybrid"],
            strategy or self._default_strategy,
        )
        query_embedding = self._embedder.embed(query)
        return self._store.search(
            query_embedding=query_embedding,
            query_text=query,
            top_k=top_k,
            strategy=strat,
            candidates_per_system=self._candidates_per_system,
        )
