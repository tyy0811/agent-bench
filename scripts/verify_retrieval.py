"""Verify retrieval quality against golden dataset.

Runs the Day 4 gate check: for each positive golden question,
does hybrid retrieval return the expected source in top-5?

Usage:
    python scripts/verify_retrieval.py
    python scripts/verify_retrieval.py --store-path .cache/store --output docs/retrieval_gate.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_bench.rag.embedder import Embedder
from agent_bench.rag.store import HybridStore


def verify(
    store_path: str = ".cache/store",
    golden_path: str = "agent_bench/evaluation/datasets/tech_docs_golden.json",
    model_name: str = "all-MiniLM-L6-v2",
    cache_dir: str = ".cache/embeddings",
    output_path: str | None = None,
) -> bool:
    store = HybridStore.load(store_path)
    embedder = Embedder(model_name=model_name, cache_dir=cache_dir)

    with open(golden_path) as f:
        questions = json.load(f)

    lines: list[str] = []
    lines.append("# Retrieval Gate Check")
    lines.append("")
    lines.append(
        f"**Store:** {store.stats().total_chunks} chunks, "
        f"{store.stats().unique_sources} sources"
    )
    lines.append("")
    lines.append("| ID | Category | Expected Source | Top-5 Sources | Recall@5 | Result |")
    lines.append("|-----|----------|----------------|---------------|----------|--------|")

    total_recall = 0.0
    scorable = 0

    for q in questions:
        qid = q["id"]
        question = q["question"]
        expected = set(q["expected_sources"])
        category = q["category"]

        vec = embedder.embed(question)
        results = store.search(vec, question, top_k=5, strategy="hybrid")
        retrieved = [r.chunk.source for r in results]
        retrieved_set = set(retrieved)

        if expected:
            recall = len(expected & retrieved_set) / len(expected)
            total_recall += recall
            scorable += 1
            result = "PASS" if recall >= 0.5 else "FAIL"
        else:
            recall = float("nan")
            result = "N/A"

        expected_str = ", ".join(sorted(expected)) if expected else "(none)"
        retrieved_str = ", ".join(dict.fromkeys(retrieved[:3]))  # dedup, first 3
        recall_str = f"{recall:.2f}" if expected else "n/a"
        lines.append(
            f"| {qid} | {category} | {expected_str} | {retrieved_str} | {recall_str} | {result} |"
        )

    avg_recall = total_recall / max(scorable, 1)
    gate_pass = avg_recall >= 0.5

    lines.append("")
    lines.append(f"**Avg Recall@5 (positive only):** {avg_recall:.2f}")
    lines.append(f"**Gate:** {'PASS' if gate_pass else 'FAIL'} (threshold >= 0.5)")

    report = "\n".join(lines)
    print(report)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report + "\n")
        print(f"\nSaved to {output_path}")

    return gate_pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify retrieval against golden dataset")
    parser.add_argument("--store-path", default=".cache/store")
    parser.add_argument("--golden-path", default="agent_bench/evaluation/datasets/tech_docs_golden.json")
    parser.add_argument("--output", default="docs/retrieval_gate.md")
    args = parser.parse_args()

    passed = verify(
        store_path=args.store_path,
        golden_path=args.golden_path,
        output_path=args.output,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
