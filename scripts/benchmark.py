"""Generate benchmark report from evaluation results.

Usage:
    python scripts/benchmark.py --results .cache/eval_results.json --output docs/benchmark_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_bench.core.config import load_config
from agent_bench.evaluation.harness import EvalResult
from agent_bench.evaluation.report import generate_report, save_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--results", default=".cache/eval_results.json")
    parser.add_argument("--output", default="docs/benchmark_report.md")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file not found at {results_path}")
        print("Run `make evaluate-fast` first to generate results.")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)
    results = [EvalResult.model_validate(r) for r in data]

    # Load config for snapshot
    config = load_config(Path(args.config) if args.config else None)
    config_dict = json.loads(config.model_dump_json())

    # Determine provider and corpus info
    provider_name = config.provider.default
    corpus_size = 16  # hardcoded for now — could read from store

    report = generate_report(
        results=results,
        config_dict=config_dict,
        provider_name=provider_name,
        corpus_size=corpus_size,
    )

    save_report(report, args.output)
    print(f"Benchmark report saved to {args.output}")
    print()
    print(report)


if __name__ == "__main__":
    main()
