"""Hybrid vector store: FAISS (dense) + BM25 (sparse) with RRF fusion."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from agent_bench.rag.chunker import Chunk


class SearchResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    chunk: Chunk
    score: float  # RRF score for hybrid, raw score for single-strategy
    rank: int
    retrieval_strategy: str


class StoreStats(BaseModel):
    total_chunks: int
    faiss_index_size: int
    unique_sources: int


def _tokenize(text: str) -> list[str]:
    """BM25 tokenizer: word chars, lowercased, punctuation stripped."""
    return re.findall(r"\w+", text.lower())


class HybridStore:
    """FAISS + BM25 hybrid store with Reciprocal Rank Fusion."""

    def __init__(self, dimension: int = 384, rrf_k: int = 60) -> None:
        self._dimension = dimension
        self._rrf_k = rrf_k
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None
        self._faiss_index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks with their pre-computed embeddings.

        Args:
            chunks: List of Chunk objects.
            embeddings: Shape (n, dimension) float32 matrix, L2-normalized.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunk count ({len(chunks)}) != embedding count ({embeddings.shape[0]})"
            )

        self._chunks.extend(chunks)

        # FAISS
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._faiss_index.add(embeddings)

        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

        # BM25 — rebuild from full corpus (BM25Okapi doesn't support incremental add)
        for chunk in chunks:
            self._tokenized_corpus.append(_tokenize(chunk.content))
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 5,
        strategy: Literal["semantic", "keyword", "hybrid"] = "hybrid",
        candidates_per_system: int = 10,
    ) -> list[SearchResult]:
        """Search the store using the specified strategy."""
        if not self._chunks:
            return []

        if strategy == "semantic":
            return self._search_semantic(query_embedding, top_k)
        elif strategy == "keyword":
            return self._search_keyword(query_text, top_k)
        elif strategy == "hybrid":
            return self._search_hybrid(query_embedding, query_text, top_k, candidates_per_system)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

    def _search_semantic(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        """Dense retrieval via FAISS."""
        k = min(top_k, len(self._chunks))
        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        scores, indices = self._faiss_index.search(query, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            results.append(
                SearchResult(
                    chunk=self._chunks[idx],
                    score=float(score),
                    rank=rank + 1,
                    retrieval_strategy="semantic",
                )
            )
        return results

    def _search_keyword(self, query_text: str, top_k: int) -> list[SearchResult]:
        """Sparse retrieval via BM25."""
        if self._bm25 is None:
            return []

        tokenized_query = _tokenize(query_text)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            results.append(
                SearchResult(
                    chunk=self._chunks[idx],
                    score=float(scores[idx]),
                    rank=rank + 1,
                    retrieval_strategy="keyword",
                )
            )
        return results

    def _search_hybrid(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int,
        candidates_per_system: int,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion of dense + sparse results."""
        dense = self._search_semantic(query_embedding, candidates_per_system)
        sparse = self._search_keyword(query_text, candidates_per_system)

        # Compute RRF scores: rrf_score(d) = Σ 1/(k + rank)
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for r in dense:
            cid = r.chunk.id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self._rrf_k + r.rank)
            chunk_map[cid] = r.chunk

        for r in sparse:
            cid = r.chunk.id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self._rrf_k + r.rank)
            chunk_map[cid] = r.chunk

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        results = []
        for rank, cid in enumerate(sorted_ids[:top_k]):
            results.append(
                SearchResult(
                    chunk=chunk_map[cid],
                    score=rrf_scores[cid],
                    rank=rank + 1,
                    retrieval_strategy="hybrid",
                )
            )
        return results

    def stats(self) -> StoreStats:
        """Return store statistics."""
        sources = {c.source for c in self._chunks}
        return StoreStats(
            total_chunks=len(self._chunks),
            faiss_index_size=self._faiss_index.ntotal,
            unique_sources=len(sources),
        )

    def save(self, path: str | Path) -> None:
        """Persist store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._faiss_index, str(path / "index.faiss"))

        with open(path / "chunks.json", "w") as f:
            json.dump([c.model_dump() for c in self._chunks], f)

        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(
                {
                    "bm25": self._bm25,
                    "tokenized_corpus": self._tokenized_corpus,
                },
                f,
            )

        if self._embeddings is not None:
            np.save(path / "embeddings.npy", self._embeddings)

    @classmethod
    def load(cls, path: str | Path, rrf_k: int = 60) -> HybridStore:
        """Load store from disk."""
        path = Path(path)

        with open(path / "chunks.json") as f:
            chunks_data = json.load(f)
        chunks = [Chunk.model_validate(d) for d in chunks_data]

        embeddings = np.load(path / "embeddings.npy")
        dimension = embeddings.shape[1]

        store = cls(dimension=dimension, rrf_k=rrf_k)
        store._chunks = chunks
        store._embeddings = embeddings

        store._faiss_index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "bm25.pkl", "rb") as f:
            bm25_data = pickle.load(f)  # noqa: S301
        store._bm25 = bm25_data["bm25"]
        store._tokenized_corpus = bm25_data["tokenized_corpus"]

        return store
