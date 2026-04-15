"""Cross-encoder reranker for improving retrieval precision."""

from __future__ import annotations

from typing import Any

import structlog

from agent_bench.rag.chunker import Chunk

log = structlog.get_logger()


class CrossEncoderReranker:
    """Rerank chunks using a cross-encoder model.

    Lazy-loads the model on first use (same pattern as Embedder).
    Accepts an optional pre-built model for testing without downloads.
    """

    def __init__(
        self,
        model: Any = None,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self._model = model
        self._model_name = model_name

    @property
    def model(self) -> Any:
        """Lazy-load the CrossEncoder on first access."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("reranker_loading", model=self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Score each (query, chunk) pair and return top_k by relevance with scores."""
        if not chunks:
            return []

        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self.model.predict(pairs)

        scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        top_results = [(chunk, float(score)) for chunk, score in scored[:top_k]]
        top_score = top_results[0][1] if top_results else 0.0

        log.info(
            "reranker_complete",
            query=query,
            input_count=len(chunks),
            output_count=len(top_results),
            top_score=top_score,
        )
        return top_results
