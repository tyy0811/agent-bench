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
from agent_bench.core.prompts import format_system_prompt
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

    # Resolve corpus-specific settings (--corpus) vs legacy single-store path.
    # Explicit --corpus routes through config.corpora[name]; without it we keep
    # the pre-multi-corpus behavior for backward compat with `make evaluate-fast`.
    if args.corpus:
        if args.corpus not in config.corpora:
            print(
                f"Error: corpus '{args.corpus}' not in config.corpora "
                f"(available: {sorted(config.corpora.keys())})"
            )
            sys.exit(1)
        corpus_cfg = config.corpora[args.corpus]
        if not corpus_cfg.available:
            print(
                f"Error: corpus '{args.corpus}' has available=false. "
                "Flip to true after data is curated and store is built."
            )
            sys.exit(1)
        if corpus_cfg.golden_dataset is None:
            print(
                f"Error: corpus '{args.corpus}' has no golden_dataset configured. "
                f"Set corpora.{args.corpus}.golden_dataset in the config."
            )
            sys.exit(1)
        store_path = corpus_cfg.store_path
        refusal_threshold = corpus_cfg.refusal_threshold
        golden_path: str = corpus_cfg.golden_dataset
        system_prompt = format_system_prompt(corpus_cfg.label)
        corpus_label = corpus_cfg.label
    else:
        task = load_task_config("tech_docs")
        store_path = config.rag.store_path
        refusal_threshold = config.rag.refusal_threshold
        golden_path = config.evaluation.golden_dataset
        system_prompt = task.system_prompt
        corpus_label = "(legacy single-store)"

    # Build the RAG pipeline
    store = HybridStore.load(store_path, rrf_k=config.rag.retrieval.rrf_k)
    embedder = Embedder(model_name=config.embedding.model, cache_dir=config.embedding.cache_dir)
    # Optional reranker
    reranker = None
    if config.rag.reranker.enabled:
        from agent_bench.rag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=config.rag.reranker.model_name)

    retriever = Retriever(
        embedder=embedder,
        store=store,
        default_strategy=config.rag.retrieval.strategy,
        candidates_per_system=config.rag.retrieval.candidates_per_system,
        reranker=reranker,
        reranker_top_k=config.rag.reranker.top_k,
    )

    # Build tools + orchestrator
    registry = ToolRegistry()
    registry.register(
        SearchTool(
            retriever=retriever,
            default_top_k=config.rag.retrieval.top_k,
            refusal_threshold=refusal_threshold,
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
    print(f"Running evaluation in '{args.mode}' mode...")
    print(f"Corpus: {corpus_label}")
    print(f"Golden dataset: {golden_path}")
    print(f"Store: {store.stats().total_chunks} chunks")
    print()

    results = await run_evaluation(
        orchestrator=orchestrator,
        system_prompt=system_prompt,
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
        "--corpus",
        default=None,
        help="Corpus name from config.corpora (e.g. 'fastapi', 'k8s'). "
        "If omitted, uses legacy rag.store_path + evaluation.golden_dataset.",
    )
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
