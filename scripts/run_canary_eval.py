"""Canary detection runner: run-judges | build-report.

run-judges scores the four dimension judges over a canary set -- each canary is
a golden question plus an injected answer with known planted defects -- and
writes a predictions JSON. build-report joins those predictions with the canary
ground truth and renders docs/_generated/canary_detection.md.

run-judges with real judges spends API budget (one judge call per canary x
dimension); it is the paid path, mirrored on `make calibrate`, and is not run in
tests or CI. build-report is free and offline. Tests drive the run by injecting
MockJudge through the build_judges seam, so no API calls happen.

Examples:
    python scripts/run_canary_eval.py run-judges \
        --canaries tests/stats/fixtures/canary/canaries.json \
        --out results/canary/predictions.json
    python scripts/run_canary_eval.py build-report \
        --canaries tests/stats/fixtures/canary/canaries.json \
        --predictions tests/stats/fixtures/canary/predictions.json
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent_bench.evaluation.judges.base import Judge

# Run as a script (python scripts/run_canary_eval.py) puts scripts/ on
# sys.path[0], not the repo root, so the repo-local stats packages are not
# importable until we add the repo root (same bootstrap as run_epochs.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stats_adapters import canary  # noqa: E402

logger = structlog.get_logger()

REPO = Path(__file__).resolve().parents[1]
DIMENSIONS = ("groundedness", "completeness", "relevance", "citation_faithfulness")
DEFAULT_OUT = REPO / "docs/_generated/canary_detection.md"

# The committed report renders from synthetic fixtures; say so in the artifact so
# no reader mistakes the simulated verdicts for a measurement. The committed-
# report test pins the artifact to this exact string, so it is the single source
# of truth for the demonstration provenance.
DEFAULT_PROVENANCE = (
    "synthetic demonstration fixtures (tests/stats/fixtures/canary); the judge "
    "verdicts are simulated, not a measurement of any real judge. Replace with a "
    "real canary set and real run-judges output for a measurement"
)


def _build_item_and_output(c: dict) -> tuple[Any, Any]:
    """Build the (GoldenQuestion, AgentResponse) pair a judge scores from one
    canary record. agent_bench imports are deferred so the offline build-report
    path imports neither agent_bench nor the embedding stack.

    ``source_chunks`` must align one-to-one with ``sources``: the citation judge
    maps a "[source: X]" claim to its evidence via ``zip(sources, source_chunks)``,
    so a missing or misaligned chunk silently scores the citation against no
    evidence. Misalignment is a loud authoring error.
    """
    from agent_bench.agents.orchestrator import AgentResponse, SourceReference
    from agent_bench.core.types import TokenUsage
    from agent_bench.evaluation.harness import GoldenQuestion

    sources = c.get("sources", [])
    source_chunks = c.get("source_chunks", [])
    if len(source_chunks) != len(sources):
        raise ValueError(
            f"canary {c['id']!r}: source_chunks must align one-to-one with sources "
            f"(one retrieved chunk per source, for the citation judge); got "
            f"{len(source_chunks)} chunks for {len(sources)} sources"
        )

    item = GoldenQuestion(
        id=c["id"],
        question=c.get("question", ""),
        expected_answer_keywords=[],
        expected_sources=[],
        category=c.get("category", "retrieval"),
        difficulty="easy",
        requires_calculator=False,
        source_snippets=c.get("source_snippets", []),
        reference_answer=c.get("reference_answer", ""),
    )
    output = AgentResponse(
        answer=c.get("answer", ""),
        sources=[SourceReference(source=s) for s in sources],
        ranked_sources=c.get("ranked_sources", []),
        source_chunks=source_chunks,
        iterations=1,
        usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
        latency_ms=0,
    )
    return item, output


async def score_canaries(canaries: list[dict], judges: dict[str, Judge]) -> list[dict[str, Any]]:
    """Score each canary on each dimension with the supplied judges
    ({dimension: Judge}). Returns one prediction dict per canary x dimension:
    {item_id, dimension, **ScoreResult}. Judges are injected, so tests pass
    MockJudge and no API call happens.
    """
    preds: list[dict[str, Any]] = []
    for c in canaries:
        item, output = _build_item_and_output(c)
        for dim in DIMENSIONS:
            result = await judges[dim].score(item, output)
            preds.append({"item_id": c["id"], "dimension": dim, **result.model_dump()})
    return preds


def build_judges(provider: str, model_id: str) -> dict[str, Judge]:
    """Construct the four real dimension judges (paid path). Seam: tests
    monkeypatch this to inject MockJudge.

    Reuses the calibration runner's rubric-loading judge factory so the canary
    judges are byte-identical to production. It is imported dynamically because
    run_calibration is a sibling script, not a library API; a static import
    would also pull its module into this script's type-check scope.
    """
    from agent_bench.core.config import load_config

    make_judge = getattr(importlib.import_module("run_calibration"), "_make_judge")
    cfg = load_config()
    return {dim: make_judge(provider, model_id, dim, cfg) for dim in DIMENSIONS}


async def cmd_run_judges(
    canaries_path: Path, out_path: Path, *, provider: str, model_id: str
) -> None:
    canaries = json.loads(canaries_path.read_text())
    judges = build_judges(provider, model_id)
    preds = await score_canaries(canaries, judges)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(preds, indent=2) + "\n")
    logger.info("canary_run_judges_complete", count=len(preds), path=str(out_path))


def cmd_build_report(
    canaries_path: Path,
    predictions_path: Path,
    out_path: Path,
    *,
    provenance: str = DEFAULT_PROVENANCE,
) -> None:
    canaries = json.loads(canaries_path.read_text())
    predictions = json.loads(predictions_path.read_text())
    md = canary.build_report(canaries, predictions, provenance=provenance)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    logger.info("canary_build_report_complete", path=str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run-judges", help="PAID: score the dimension judges over a canary set")
    p_run.add_argument("--canaries", type=Path, required=True)
    p_run.add_argument("--out", type=Path, required=True)
    p_run.add_argument("--provider", default="anthropic")
    p_run.add_argument("--model-id", default="claude-haiku-4-5")

    p_rep = sub.add_parser(
        "build-report", help="FREE: render the detection report from predictions"
    )
    p_rep.add_argument("--canaries", type=Path, required=True)
    p_rep.add_argument("--predictions", type=Path, required=True)
    p_rep.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p_rep.add_argument("--provenance", default=DEFAULT_PROVENANCE)

    args = parser.parse_args()
    if args.cmd == "run-judges":
        asyncio.run(
            cmd_run_judges(args.canaries, args.out, provider=args.provider, model_id=args.model_id)
        )
    elif args.cmd == "build-report":
        cmd_build_report(args.canaries, args.predictions, args.out, provenance=args.provenance)


if __name__ == "__main__":
    main()
