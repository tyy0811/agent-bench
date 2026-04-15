"""Ingest documents into the hybrid vector store.

Usage:
    python scripts/ingest.py --config configs/tasks/tech_docs.yaml
    python scripts/ingest.py --doc-dir data/tech_docs/ --store-path .cache/store
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the package is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_bench.rag.chunker import chunk_text
from agent_bench.rag.embedder import Embedder
from agent_bench.rag.store import HybridStore


def ingest(
    doc_dir: str,
    store_path: str,
    chunk_strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    model_name: str = "all-MiniLM-L6-v2",
    cache_dir: str = ".cache/embeddings",
) -> None:
    """Ingest all markdown files from doc_dir into a HybridStore."""
    doc_path = Path(doc_dir)
    if not doc_path.exists():
        print(f"Error: document directory {doc_dir} does not exist")
        sys.exit(1)

    # Exclude curation metadata files that live alongside corpus content.
    # SOURCES.md and QUESTION_PLAN.md are version-controlled curation
    # artifacts, not corpus content.
    _EXCLUDED = {"SOURCES.md", "QUESTION_PLAN.md", "README.md"}
    md_files = sorted(f for f in doc_path.glob("*.md") if f.name not in _EXCLUDED)
    if not md_files:
        print(f"Error: no markdown files found in {doc_dir}")
        sys.exit(1)

    print(f"Found {len(md_files)} markdown files in {doc_dir}")

    # Chunk all documents
    all_chunks = []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        source = md_file.name  # bare filename
        chunks = chunk_text(
            text, source, strategy=chunk_strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        print(f"  {source}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    # Embed
    print(f"Embedding with {model_name}...")
    embedder = Embedder(model_name=model_name, cache_dir=cache_dir)
    texts = [c.content for c in all_chunks]
    embeddings = embedder.embed_batch(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Store
    store = HybridStore(dimension=embeddings.shape[1])
    store.add(all_chunks, embeddings)
    store.save(store_path)

    stats = store.stats()
    print(f"Store saved to {store_path}")
    print(f"  Chunks: {stats.total_chunks}")
    print(f"  FAISS index size: {stats.faiss_index_size}")
    print(f"  Unique sources: {stats.unique_sources}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument("--doc-dir", default="data/tech_docs/", help="Document directory")
    parser.add_argument("--store-path", default=".cache/store", help="Store output path")
    parser.add_argument("--chunk-strategy", default="recursive", choices=["recursive", "fixed"])
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--cache-dir", default=".cache/embeddings", help="Embedding cache dir")
    parser.add_argument(
        "--config", default=None, help="Task config YAML (overrides other args for doc-dir)"
    )
    args = parser.parse_args()

    doc_dir = args.doc_dir
    if args.config:
        from agent_bench.core.config import load_task_config

        task = load_task_config(Path(args.config).stem, path=Path(args.config))
        doc_dir = task.document_dir

    ingest(
        doc_dir=doc_dir,
        store_path=args.store_path,
        chunk_strategy=args.chunk_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
