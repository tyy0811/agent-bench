"""Embedding wrapper around sentence-transformers with disk cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


class Embedder:
    """Embeds text using sentence-transformers with optional disk cache.

    Accepts any object with an encode() method so tests can inject a mock
    without downloading the 80MB model.
    """

    def __init__(
        self,
        model: Any = None,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = ".cache/embeddings",
    ) -> None:
        if model is not None:
            self._model = model
        else:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns shape (384,) normalized vector."""
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        cache_path = self._cache_dir / f"{cache_key}.npy"

        if cache_path.exists():
            vec = np.load(cache_path)
            return np.asarray(vec, dtype=np.float32)

        vec = self._model.encode([text], normalize_embeddings=True)[0]
        vec = np.asarray(vec, dtype=np.float32)
        np.save(cache_path, vec)
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Returns shape (n, 384) normalized matrix."""
        results = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            cache_key = hashlib.sha256(text.encode()).hexdigest()
            cache_path = self._cache_dir / f"{cache_key}.npy"
            if cache_path.exists():
                results.append((i, np.load(cache_path)))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                results.append((i, None))

        if uncached_texts:
            vecs = self._model.encode(uncached_texts, normalize_embeddings=True)
            vecs = np.asarray(vecs, dtype=np.float32)
            for j, idx in enumerate(uncached_indices):
                vec = vecs[j]
                # Save to cache
                cache_key = hashlib.sha256(uncached_texts[j].encode()).hexdigest()
                cache_path = self._cache_dir / f"{cache_key}.npy"
                np.save(cache_path, vec)
                # Update results
                for k, (ri, rv) in enumerate(results):
                    if ri == idx:
                        results[k] = (ri, vec)
                        break

        # Sort by original index and stack
        results.sort(key=lambda x: x[0])
        return np.stack([r[1] for r in results])  # type: ignore[misc]
