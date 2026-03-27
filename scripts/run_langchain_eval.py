"""Run LangChain baseline evaluation against the golden dataset.

Usage:
    python scripts/run_langchain_eval.py --provider openai
    python scripts/run_langchain_eval.py --provider anthropic
    python scripts/run_langchain_eval.py --provider openai --max-questions 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_bench.core.config import load_config, load_task_config
from agent_bench.evaluation.report import generate_report, save_report
from agent_bench.langchain_baseline.agent import create_langchain_agent
from agent_bench.langchain_baseline.retriever import AgentBenchRetriever
from agent_bench.langchain_baseline.runner import run_langchain_evaluation
from agent_bench.langchain_baseline.tools import LangChainSearchTool, create_calculator_tool
from agent_bench.rag.embedder import Embedder
from agent_bench.rag.retriever import Retriever
from agent_bench.rag.store import HybridStore


async def main_async(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config) if args.config else None)
    task = load_task_config("tech_docs")

    # Build existing RAG pipeline (same as scripts/evaluate.py)
    store = HybridStore.load(config.rag.store_path, rrf_k=config.rag.retrieval.rrf_k)
    embedder = Embedder(model_name=config.embedding.model, cache_dir=config.embedding.cache_dir)

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

    # Wrap in LangChain components
    lc_retriever = AgentBenchRetriever(retriever=retriever, top_k=config.rag.retrieval.top_k)
    search_tool = LangChainSearchTool(lc_retriever)
    calc_tool = create_calculator_tool()

    agent_executor = create_langchain_agent(
        tools=[search_tool.as_tool(), calc_tool],
        provider=args.provider,
        system_prompt=task.system_prompt,
    )

    # Run evaluation
    golden_path = config.evaluation.golden_dataset
    print("Running LangChain baseline evaluation...")
    print(f"  Provider:  {args.provider}")
    print(f"  Store:     {store.stats().total_chunks} chunks")
    print(f"  Golden:    {golden_path}")
    if args.max_questions:
        print(f"  Limit:     {args.max_questions} questions")
    print()

    results = await run_langchain_evaluation(
        agent_executor=agent_executor,
        search_tool_state=search_tool,
        golden_path=golden_path,
        provider_name=args.provider,
        max_questions=args.max_questions,
    )

    # Save raw results JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_data = [r.model_dump() for r in results]
    output_path.write_text(json.dumps(results_data, indent=2, default=str))
    print(f"Results JSON: {output_path}")

    # Generate markdown report (reuses existing report generator)
    report = generate_report(
        results,
        provider_name=f"langchain-{args.provider}",
        corpus_size=store.stats().unique_sources,
    )
    report_path = Path(f"docs/langchain_benchmark_{args.provider}.md")
    save_report(report, report_path)
    print(f"Report:      {report_path}")

    # Print summary
    positive = [r for r in results if r.category != "out_of_scope"]
    errors = [r for r in results if r.answer.startswith("ERROR")]
    avg_p5 = sum(r.retrieval_precision for r in positive) / max(len(positive), 1)
    avg_r5 = sum(r.retrieval_recall for r in positive) / max(len(positive), 1)
    avg_khr = sum(r.keyword_hit_rate for r in positive) / max(len(positive), 1)
    avg_lat = sum(r.latency_ms for r in results) / max(len(results), 1)

    print(f"\nSummary ({len(results)} questions, {len(errors)} errors):")
    print(f"  Avg P@5:     {avg_p5:.2f}")
    print(f"  Avg R@5:     {avg_r5:.2f}")
    print(f"  Avg KHR:     {avg_khr:.2f}")
    print(f"  Avg latency: {avg_lat:,.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangChain baseline evaluation")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
    )
    parser.add_argument("--config", default=None, help="Config YAML path")
    parser.add_argument("--output", default=".cache/langchain_eval_results.json")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions (for testing)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
