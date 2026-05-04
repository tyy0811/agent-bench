"""Calibration runner: generate-outputs | run-judges | build-table.

Orchestrates Steps A, C, D from the design doc's data flow. Step B
(hand-labeling) is manual — done in a Jupyter notebook reading
results/calibration_v1_system_outputs.json and appending to
measurements/2026-05-04-judge-calibration-labels.jsonl.

Examples:
    python scripts/run_calibration.py generate-outputs --concurrency 5
    python scripts/run_calibration.py run-judges --row-config=configs/calibration/rows/baseline.yaml
    python scripts/run_calibration.py build-table
    python scripts/run_calibration.py build-table --strict
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import structlog
import yaml

logger = structlog.get_logger()

REPO = Path(__file__).resolve().parents[1]
CALIBRATION_SPEC = REPO / "agent_bench/evaluation/datasets/calibration_v1.json"
SYSTEM_OUTPUTS = REPO / "results/calibration_v1_system_outputs.json"
LABELS_PATH = REPO / "measurements/2026-05-04-judge-calibration-labels.jsonl"
KAPPA_TABLE_OUT = REPO / "docs/_generated/kappa_table.md"


def _resolve_concurrency(cli_value: int | None) -> int:
    """CLI flag overrides config field; default is 5. Logs the resolved value."""
    if cli_value is not None:
        resolved = cli_value
    else:
        cfg_path = REPO / "configs/default.yaml"
        cfg_concurrency = None
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            cfg_concurrency = (cfg.get("evaluation", {}) or {}).get(
                "calibration_concurrency"
            )
        resolved = cfg_concurrency if cfg_concurrency is not None else 5
    logger.info("calibration_concurrency_resolved", value=resolved)
    return resolved


# --- Subcommand: generate-outputs (Step A) ---


async def cmd_generate_outputs(concurrency: int) -> None:
    """Run the orchestrator against the 30 calibration items with a frozen
    configuration; write results/calibration_v1_system_outputs.json.
    """
    from agent_bench.agents.orchestrator import Orchestrator
    from agent_bench.core.config import load_config
    from agent_bench.core.provider import AnthropicProvider
    from agent_bench.evaluation.harness import load_golden_dataset
    from agent_bench.tools.registry import build_default_registry

    spec = json.loads(CALIBRATION_SPEC.read_text())
    target_ids = {i["id"]: i for i in spec["items"]}

    fastapi = load_golden_dataset(
        REPO / "agent_bench/evaluation/datasets/tech_docs_golden.json"
    )
    k8s = load_golden_dataset(
        REPO / "agent_bench/evaluation/datasets/k8s_golden.json"
    )
    items = [q for q in (fastapi + k8s) if q.id in target_ids]
    if len(items) != len(target_ids):
        missing = set(target_ids) - {q.id for q in items}
        raise SystemExit(
            f"calibration items not found in goldens: {sorted(missing)}"
        )

    cfg = load_config()
    provider = AnthropicProvider(cfg)
    registry = build_default_registry(cfg)
    orchestrator = Orchestrator(provider=provider, registry=registry)

    sem = asyncio.Semaphore(concurrency)

    async def _run_one(item):
        async with sem:
            response = await orchestrator.run(
                question=item.question,
                system_prompt="You are a helpful assistant.",
            )
            answer = response.answer
            sources = sorted(s.source for s in response.sources)
            sys_hash = hashlib.sha256(
                f"{item.id}\x00{answer}\x00{','.join(sources)}".encode("utf-8")
            ).hexdigest()
            return {
                "item_id": item.id,
                "question": item.question,
                "category": item.category,
                "answer": answer,
                "sources": [s.source for s in response.sources],
                "ranked_sources": response.ranked_sources,
                "source_chunks": response.source_chunks,
                "source_snippets": item.source_snippets,
                "reference_answer": item.reference_answer,
                "system_output_hash": sys_hash,
                "stratum": target_ids[item.id]["stratum"],
                "corpus": target_ids[item.id]["corpus"],
            }

    records = await asyncio.gather(*[_run_one(it) for it in items])
    SYSTEM_OUTPUTS.parent.mkdir(parents=True, exist_ok=True)
    SYSTEM_OUTPUTS.write_text(json.dumps(records, indent=2) + "\n")
    logger.info(
        "generate_outputs_complete", count=len(records), path=str(SYSTEM_OUTPUTS)
    )


# --- Subcommand: run-judges (Step C, one row per invocation) ---


def _make_provider(name: str, cfg):
    from agent_bench.core.provider import AnthropicProvider, OpenAIProvider

    if name == "anthropic":
        return AnthropicProvider(cfg)
    if name == "openai":
        return OpenAIProvider(cfg)
    raise ValueError(f"unknown provider: {name}")


def _make_judge(provider_name: str, model_id: str, dimension: str, cfg):
    from agent_bench.evaluation.judges.base import Rubric
    from agent_bench.evaluation.judges.citation_faithfulness import (
        CitationFaithfulnessJudge,
    )
    from agent_bench.evaluation.judges.completeness import CompletenessJudge
    from agent_bench.evaluation.judges.groundedness import GroundednessJudge
    from agent_bench.evaluation.judges.relevance import RelevanceJudge

    judge_class = {
        "groundedness": GroundednessJudge,
        "relevance": RelevanceJudge,
        "completeness": CompletenessJudge,
        "citation_faithfulness": CitationFaithfulnessJudge,
    }
    rubric_dir = REPO / "agent_bench/evaluation/rubrics"
    rubric = Rubric.from_markdown_file(rubric_dir / f"{dimension}.md")
    return judge_class[dimension](
        judge_provider=_make_provider(provider_name, cfg),
        rubric=rubric,
        model_id=model_id,
    )


def _build_item_and_output(rec: dict):
    from agent_bench.agents.orchestrator import AgentResponse, SourceReference
    from agent_bench.core.types import TokenUsage
    from agent_bench.evaluation.harness import GoldenQuestion

    item = GoldenQuestion(
        id=rec["item_id"],
        question=rec["question"],
        expected_answer_keywords=[],
        expected_sources=[],
        category=rec["category"],
        difficulty="easy",
        requires_calculator=False,
        source_snippets=rec.get("source_snippets", []),
        reference_answer=rec.get("reference_answer", ""),
    )
    output = AgentResponse(
        answer=rec["answer"],
        sources=[SourceReference(source=s) for s in rec["sources"]],
        ranked_sources=rec.get("ranked_sources", []),
        source_chunks=rec.get("source_chunks", []),
        iterations=1,
        usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
        latency_ms=0,
    )
    return item, output


async def cmd_run_judges(row_config_path: Path, concurrency: int) -> None:
    """Score the frozen system outputs with the row's judge configuration."""
    from agent_bench.core.config import load_config
    from agent_bench.evaluation.variance.jury import jury
    from agent_bench.evaluation.variance.rubric_permute import rubric_permute

    if not SYSTEM_OUTPUTS.exists():
        raise SystemExit(
            f"{SYSTEM_OUTPUTS} not found — run `generate-outputs` first."
        )
    row = yaml.safe_load(row_config_path.read_text())
    outputs = json.loads(SYSTEM_OUTPUTS.read_text())

    cfg = load_config()
    sem = asyncio.Semaphore(concurrency)
    all_results: list[dict] = []
    strategy = row["strategy"]

    def _skip_oos(rec: dict, dim: str) -> bool:
        return rec["category"] == "out_of_scope" and dim != "relevance"

    if strategy == "single":
        # Build one judge per dimension up-front, then gather all
        # (dim, item) pairs in a single asyncio.gather call. Previous
        # design serialized across dimensions (each dim awaited fully
        # before the next started), leaving Phase-11 wall-clock on the
        # table when the calibration spend is API-rate-limited.
        judges_by_dim = {
            dim: _make_judge(row["provider"], row["model_id"], dim, cfg)
            for dim in row["dimensions"]
        }

        async def score_one(rec: dict, dim: str, judge):
            async with sem:
                if _skip_oos(rec, dim):
                    return None
                item, output = _build_item_and_output(rec)
                result = await judge.score(item, output)
                return {"dimension": dim, **result.model_dump()}

        coros = [
            score_one(rec, dim, judge)
            for dim, judge in judges_by_dim.items()
            for rec in outputs
        ]
        gathered = await asyncio.gather(*coros)
        all_results.extend([r for r in gathered if r is not None])

    elif strategy == "rubric_permute":
        # Sequential per-item by design: PermutedJudge writes to the
        # sidecar JSONL with append mode and within-call ordering matters
        # for downstream per-permutation analysis (the kappa_table joins
        # by item_id but the sidecar order encodes the permutation seed
        # sequence). Across-dim parallelism is left for v1.1 once the
        # sidecar contract proves stable.
        for dim in row["dimensions"]:
            judge = _make_judge(row["provider"], row["model_id"], dim, cfg)
            sidecar = REPO / row.get(
                "sidecar_path", "results/calibration_v1_permute_members.jsonl"
            )
            permuted = rubric_permute(
                judge,
                n=row["options"]["n_permutations"],
                seeds=row["options"]["seeds"],
                sidecar_path=sidecar,
            )
            for rec in outputs:
                if _skip_oos(rec, dim):
                    continue
                item, output = _build_item_and_output(rec)
                result = await permuted.score(item, output)
                all_results.append({"dimension": dim, **result.model_dump()})

    elif strategy == "jury":
        # Same sequential rationale as rubric_permute: jury writes a
        # per-member sidecar and downstream analysis benefits from stable
        # ordering. The asyncio.gather inside Jury.score does parallelize
        # member calls within an item; the across-item / across-dim
        # serialization is the conservative choice.
        for dim in row["dimensions"]:
            members = [
                _make_judge(m["provider"], m["model_id"], dim, cfg)
                for m in row["members"]
            ]
            sidecar = REPO / row["sidecar_path"]
            weights = (
                _load_weights_from_baseline(REPO / row["weights_source"], dim)
                if row.get("aggregation") == "kappa_weighted"
                else None
            )
            j = jury(
                judges=members,
                aggregation=row["aggregation"],
                weights=weights,
                quorum=row.get("quorum"),
                sidecar_path=sidecar,
            )
            for rec in outputs:
                if _skip_oos(rec, dim):
                    continue
                item, output = _build_item_and_output(rec)
                result = await j.score(item, output)
                all_results.append({"dimension": dim, **result.model_dump()})
    else:
        raise SystemExit(f"unknown strategy: {strategy}")

    out_path = REPO / row["output_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2) + "\n")
    logger.info(
        "run_judges_complete",
        row=row["label"],
        count=len(all_results),
        path=str(out_path),
    )


def _load_weights_from_baseline(
    baseline_path: Path, dimension: str
) -> dict[str, float]:
    """Compute per-judge weight = κ vs labels for the dimension, from baseline run.

    Stub for v1: returns equal weights (1.0 for each judge_id seen in
    the baseline file). Replaced by real κ-derived weights once labels
    + baseline are both populated. Documented in writeup as caveat:
    'weights estimated on calibration set; production deployment would
    use a held-out validation set'.
    """
    if not baseline_path.exists():
        logger.warning(
            "weights_source_missing",
            path=str(baseline_path),
            fallback="equal_weights",
        )
        return {}
    baseline = json.loads(baseline_path.read_text())
    judge_ids = {
        r["judge_id"] for r in baseline if r.get("dimension") == dimension
    }
    return {jid: 1.0 for jid in judge_ids}


# --- Subcommand: build-table (Step D) ---


def cmd_build_table(strict: bool) -> None:
    from agent_bench.evaluation.calibration.report import generate_kappa_table

    predictions_glob = str(REPO / "results/calibration_v1_judge_*.json")
    generate_kappa_table(
        predictions_glob=predictions_glob,
        labels_path=str(LABELS_PATH),
        output_path=str(KAPPA_TABLE_OUT),
        strict=strict,
    )
    logger.info("build_table_complete", path=str(KAPPA_TABLE_OUT), strict=strict)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser(
        "generate-outputs", help="Step A: generate frozen system outputs"
    )
    p_gen.add_argument("--concurrency", type=int, default=None)

    p_run = sub.add_parser("run-judges", help="Step C: score one ablation row")
    p_run.add_argument("--row-config", type=Path, required=True)
    p_run.add_argument("--concurrency", type=int, default=None)

    p_tab = sub.add_parser(
        "build-table", help="Step D: aggregate predictions into κ table"
    )
    p_tab.add_argument(
        "--strict",
        action="store_true",
        help="Raise on missing predictions/labels (final-artifact path)",
    )

    args = parser.parse_args()
    if args.cmd == "generate-outputs":
        asyncio.run(cmd_generate_outputs(_resolve_concurrency(args.concurrency)))
    elif args.cmd == "run-judges":
        asyncio.run(
            cmd_run_judges(args.row_config, _resolve_concurrency(args.concurrency))
        )
    elif args.cmd == "build-table":
        cmd_build_table(strict=args.strict)


if __name__ == "__main__":
    main()
