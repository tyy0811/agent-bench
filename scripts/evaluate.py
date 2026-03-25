"""Run the evaluation harness.

Usage:
    python scripts/evaluate.py --mode deterministic
    python scripts/evaluate.py --mode full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_bench.agents.orchestrator import Orchestrator
from agent_bench.core.config import load_config, load_task_config
from agent_bench.core.provider import MockProvider, create_provider
from agent_bench.evaluation.harness import run_evaluation
from agent_bench.rag.embedder import Embedder
from agent_bench.rag.retriever import Retriever
from agent_bench.rag.store import HybridStore
from agent_bench.tools.calculator import CalculatorTool
from agent_bench.tools.registry import ToolRegistry
from agent_bench.tools.search import SearchTool


async def main_async(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config) if args.config else None)
    task = load_task_config("tech_docs")

    # Build the RAG pipeline
    store = HybridStore.load(config.rag.store_path, rrf_k=config.rag.retrieval.rrf_k)
    embedder = Embedder(model_name=config.embedding.model, cache_dir=config.embedding.cache_dir)
    retriever = Retriever(
        embedder=embedder,
        store=store,
        default_strategy=config.rag.retrieval.strategy,
        candidates_per_system=config.rag.retrieval.candidates_per_system,
    )

    # Build tools + orchestrator
    registry = ToolRegistry()
    registry.register(
        SearchTool(
            retriever=retriever,
            default_top_k=config.rag.retrieval.top_k,
            refusal_threshold=config.rag.refusal_threshold,
        )
    )
    registry.register(CalculatorTool())

    provider = create_provider(config)
    orchestrator = Orchestrator(
        provider=provider,
        registry=registry,
        max_iterations=config.agent.max_iterations,
        temperature=config.agent.temperature,
    )

    # Judge provider for full mode — uses configured judge_provider
    judge = None
    if args.mode == "full":
        from agent_bench.core.config import AppConfig, ProviderConfig

        judge_config = AppConfig(
            provider=ProviderConfig(
                default=config.evaluation.judge_provider,
                models=config.provider.models,
            )
        )
        judge = create_provider(judge_config)

    # Run evaluation
    golden_path = config.evaluation.golden_dataset
    print(f"Running evaluation in '{args.mode}' mode...")
    print(f"Golden dataset: {golden_path}")
    print(f"Store: {store.stats().total_chunks} chunks")
    print()

    results = await run_evaluation(
        orchestrator=orchestrator,
        system_prompt=task.system_prompt,
        golden_path=golden_path,
        judge_provider=judge,
    )

    # Save results as JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_data = [r.model_dump() for r in results]
    output_path.write_text(json.dumps(results_data, indent=2, default=str))
    print(f"Results saved to {output_path}")

    # Print summary
    positive = [r for r in results if r.category != "out_of_scope"]
    avg_p5 = sum(r.retrieval_precision for r in positive) / max(len(positive), 1)
    avg_r5 = sum(r.retrieval_recall for r in positive) / max(len(positive), 1)
    avg_khr = sum(r.keyword_hit_rate for r in positive) / max(len(positive), 1)
    print(f"\nSummary ({len(results)} questions):")
    print(f"  Avg P@5:  {avg_p5:.2f}")
    print(f"  Avg R@5:  {avg_r5:.2f}")
    print(f"  Avg KHR:  {avg_khr:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("--config", default=None, help="Config YAML path")
    parser.add_argument(
        "--mode",
        choices=["deterministic", "full"],
        default="deterministic",
    )
    parser.add_argument("--output", default=".cache/eval_results.json")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
