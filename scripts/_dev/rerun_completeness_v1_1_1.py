"""Plan 3A follow-through: re-run gpt-4o-mini on completeness for all 26
calibration items with the v1.1.1 recency-positioned paraphrase clause now
permanent in CompletenessJudge.

Methodological note: only gpt-4o-mini is re-run. Haiku stays as control —
its v1.1 completeness predictions remain valid. This makes the v1.1.1
delta cleanly attributable to the intervention's effect on the affected
judge, not a confound from re-prompting both judges.

Outputs:
  - results/calibration_v1_judge_jury_kappa_weighted_v1_1_1_members.jsonl
    (Haiku rows copied from v1.1 sidecar; gpt-4o-mini rows fresh)
  - results/calibration_v1_judge_jury_kappa_weighted_v1_1_1.json
    (re-aggregated jury verdicts using fresh gpt-4o-mini + existing Haiku)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_bench.agents.orchestrator import AgentResponse, SourceReference  # noqa: E402
from agent_bench.core.provider import OpenAIProvider  # noqa: E402
from agent_bench.core.types import TokenUsage  # noqa: E402
from agent_bench.evaluation.harness import GoldenQuestion  # noqa: E402
from agent_bench.evaluation.judges.base import Rubric  # noqa: E402
from agent_bench.evaluation.judges.completeness import CompletenessJudge  # noqa: E402

LABELS = REPO / "measurements/2026-05-04-judge-calibration-labels.jsonl"
SIDECAR_V1_1 = REPO / "results/calibration_v1_judge_jury_kappa_weighted_members.jsonl"
SYSTEM_OUTPUTS = REPO / "results/calibration_v1_system_outputs.json"
NEW_SIDECAR = REPO / "results/calibration_v1_judge_jury_kappa_weighted_v1_1_1_members.jsonl"


def _build_item_and_output(rec: dict) -> tuple[GoldenQuestion, AgentResponse]:
    item = GoldenQuestion(
        id=rec["item_id"],
        question=rec.get("question", ""),
        expected_answer_keywords=[],
        expected_sources=[],
        category=rec.get("category", "retrieval"),
        difficulty="easy",
        requires_calculator=False,
        reference_answer=rec.get("reference_answer", ""),
        source_snippets=rec.get("source_snippets", []),
    )
    output = AgentResponse(
        answer=rec["answer"],
        sources=[SourceReference(source=s) for s in rec.get("sources", [])],
        iterations=1,
        usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0.0),
        latency_ms=0,
    )
    return item, output


async def main() -> None:
    rubric = Rubric.from_markdown_file(
        REPO / "agent_bench/evaluation/rubrics/completeness.md"
    )
    outputs = json.loads(SYSTEM_OUTPUTS.read_text())
    by_id = {r["item_id"]: r for r in outputs}

    # Load existing Haiku completeness rows from v1.1 sidecar (control).
    haiku_completeness_rows: list[dict] = []
    by_hash_latest: dict[tuple[str, str], dict] = {}
    for line in SIDECAR_V1_1.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by_hash_latest[(rec["judge_id"], rec["system_output_hash"])] = rec

    for rec in by_hash_latest.values():
        if "haiku" in rec["judge_id"].lower() and rec["judge_id"].endswith("_completeness"):
            haiku_completeness_rows.append(rec)

    # Run gpt-4o-mini CompletenessJudge with the v1.1.1 prompt on all items
    # that have a system output (= 30 items).
    provider = OpenAIProvider(model="gpt-4o-mini-2024-07-18")
    judge = CompletenessJudge(
        judge_provider=provider, rubric=rubric, model_id="gpt-4o-mini-2024-07-18"
    )

    print(f"Running gpt-4o-mini CompletenessJudge (v1.1.1 prompt) on {len(outputs)} items")
    fresh_gpt_rows: list[dict] = []
    for rec in outputs:
        item, output = _build_item_and_output(rec)
        result = await judge.score(item, output)
        row = result.model_dump()
        row["item_id"] = item.id
        fresh_gpt_rows.append(row)
        score_marker = result.score
        print(f"  {item.id:<10} score={score_marker} cost=${result.cost_usd:.4f}")

    total_cost = sum(r["cost_usd"] for r in fresh_gpt_rows)
    print(f"\nTotal cost: ${total_cost:.4f}")

    # Write the v1.1.1 sidecar: Haiku completeness rows (unchanged from v1.1)
    # + fresh gpt-4o-mini completeness rows.
    with NEW_SIDECAR.open("w") as f:
        for r in haiku_completeness_rows:
            f.write(json.dumps(r) + "\n")
        for r in fresh_gpt_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(haiku_completeness_rows)} Haiku + {len(fresh_gpt_rows)} GPT rows to {NEW_SIDECAR}")


if __name__ == "__main__":
    asyncio.run(main())
