"""Retrieval orchestration: embed query → search store → rerank → return."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from agent_bench.rag.embedder import Embedder
from agent_bench.rag.store import HybridStore, SearchResult

if TYPE_CHECKING:
    from agent_bench.rag.reranker import CrossEncoderReranker


@dataclass
class RetrievalResult:
    """Retriever output with metadata for stage events."""
    results: list[SearchResult] = field(default_factory=list)
    pre_rerank_count: int = 0


class Retriever:
    """Thin glue between embedder, store, and optional reranker."""

    def __init__(
        self,
        embedder: Embedder,
        store: HybridStore,
        default_strategy: Literal["semantic", "keyword", "hybrid"] = "hybrid",
        candidates_per_system: int = 10,
        reranker: CrossEncoderReranker | None = None,
        reranker_top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._default_strategy = default_strategy
        self._candidates_per_system = candidates_per_system
        self._reranker = reranker
        self._reranker_top_k = reranker_top_k

    async def search(
        self,
        query: str,
        top_k: int = 5,
        strategy: str | None = None,
    ) -> RetrievalResult:
        """Embed query, search store, optionally rerank."""
        strat: Literal["semantic", "keyword", "hybrid"] = cast(
            Literal["semantic", "keyword", "hybrid"],
            strategy or self._default_strategy,
        )
        query_embedding = self._embedder.embed(query)

        # When reranker is enabled, fetch all RRF-fused candidates
        # and let the reranker handle truncation
        store_top_k = self._candidates_per_system * 2 if self._reranker else top_k

        results = self._store.search(
            query_embedding=query_embedding,
            query_text=query,
            top_k=store_top_k,
            strategy=strat,
            candidates_per_system=self._candidates_per_system,
        )

        pre_rerank_count = len(results)

        if self._reranker and results:
            # Preserve original RRF scores — the refusal gate needs them
            rrf_scores = {r.chunk.id: r.score for r in results}

            chunks = [r.chunk for r in results]
            reranked = self._reranker.rerank(
                query, chunks, top_k=self._reranker_top_k,
            )
            # Rebuild SearchResult objects with new ranks but original RRF scores
            results = [
                SearchResult(
                    chunk=chunk,
                    score=rrf_scores.get(chunk.id, 0.0),
                    rank=rank + 1,
                    retrieval_strategy="hybrid+reranker",
                    rerank_score=rerank_score,
                )
                for rank, (chunk, rerank_score) in enumerate(reranked)
            ]
        else:
            pre_rerank_count = 0  # no reranking happened

        return RetrievalResult(results=results, pre_rerank_count=pre_rerank_count)
