"""Document chunking with recursive and fixed-size strategies."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    content: str
    source: str  # bare filename, e.g. "fastapi_path_params.md"
    chunk_index: int
    metadata: dict = Field(default_factory=dict)


def _make_chunk_id(content: str, source: str) -> str:
    """Deterministic ID from content + source."""
    return hashlib.sha256(f"{source}:{content}".encode()).hexdigest()[:16]


def chunk_recursive(
    text: str,
    source: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split text on paragraph → newline → sentence → space boundaries.

    Tries the coarsest separator first; only falls through to finer
    separators when a segment still exceeds chunk_size.
    """
    separators = ["\n\n", "\n", ". ", " "]
    segments = _recursive_split(text, separators, chunk_size)
    return _segments_to_chunks(segments, source, chunk_size, chunk_overlap)


def chunk_fixed(
    text: str,
    source: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split text into fixed-size character windows with overlap."""
    if not text.strip():
        return []
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        content = text[start:end].strip()
        if content:
            chunks.append(
                Chunk(
                    id=_make_chunk_id(content, source),
                    content=content,
                    source=source,
                    chunk_index=idx,
                    metadata={"strategy": "fixed"},
                )
            )
            idx += 1
        step = chunk_size - chunk_overlap
        if step <= 0:
            step = 1
        start += step
    return chunks


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
) -> list[str]:
    """Recursively split text using progressively finer separators."""
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        # No separators left — hard split
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep = separators[0]
    remaining_seps = separators[1:]
    parts = text.split(sep)

    segments: list[str] = []
    current = ""

    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                segments.append(current)
            if len(part) > chunk_size:
                segments.extend(_recursive_split(part, remaining_seps, chunk_size))
            else:
                current = part
                continue
            current = ""

    if current.strip():
        segments.append(current)

    return segments


def _segments_to_chunks(
    segments: list[str],
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Convert text segments to Chunk objects, applying overlap where possible."""
    chunks: list[Chunk] = []
    for idx, seg in enumerate(segments):
        content = seg.strip()
        if not content:
            continue
        chunks.append(
            Chunk(
                id=_make_chunk_id(content, source),
                content=content,
                source=source,
                chunk_index=idx,
                metadata={"strategy": "recursive"},
            )
        )
    return chunks


Strategy = Literal["recursive", "fixed"]


def chunk_text(
    text: str,
    source: str,
    strategy: Strategy = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Chunk text using the specified strategy."""
    if strategy == "recursive":
        return chunk_recursive(text, source, chunk_size, chunk_overlap)
    elif strategy == "fixed":
        return chunk_fixed(text, source, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
