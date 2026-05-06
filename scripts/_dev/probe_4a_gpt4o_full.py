"""Plan 4A probe: GPT-4o (full, not mini) on 5 of the 14 v1.1.1 unchanged
items, using the v1.1.1 production prompt (paraphrase recency clause
included).

Items (gold=2/pred=1 unchanged after v1.1.1 intervention):
  - k8s_006, k8s_018  — the 2/5 that didn't shift in the 3A 5-item probe.
                         We already have GPT-4o-mini's reasoning on these
                         WITH the intervention; GPT-4o on the same prompt
                         is a clean A/B at fixed prompt, varying model.
  - q011, q012        — fastapi residuals.
  - k8s_001           — k8s residual where Haiku also disagreed (Haiku
                         scored 1, gold 2).

Diagnostic question: does a stronger model handle the residual at the
same v1.1.1 prompt?

  - GPT-4o scores 2 on most → residual is small-model-specific;
    v1.2 fix #3 (per-dim exclusion / stronger model on completeness)
    gets clean empirical support.
  - GPT-4o also scores 1 → rubric is under-specified for whatever
    failure mode these items hit; v1.2 needs additional rubric anchoring,
    not just judge-membership tuning.

Run:
    OPENAI_API_KEY=... python scripts/_dev/probe_4a_gpt4o_full.py
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

ITEMS = ["k8s_006", "k8s_018", "q011", "q012", "k8s_001"]
GPT4O_FULL = "gpt-4o-2024-08-06"

# Prior scores (gpt-4o-mini under v1.1.1 prompt, full-26 re-run output)
PRIOR_GPT4O_MINI_V1_1_1 = {iid: 1 for iid in ITEMS}
GOLD = {iid: 2 for iid in ITEMS}


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
    outputs = json.loads(
        (REPO / "results/calibration_v1_system_outputs.json").read_text()
    )
    by_id = {r["item_id"]: r for r in outputs}

    provider = OpenAIProvider(model=GPT4O_FULL)
    judge = CompletenessJudge(
        judge_provider=provider, rubric=rubric, model_id=GPT4O_FULL
    )

    print("=" * 80)
    print(f"Plan 4A — GPT-4o full ({GPT4O_FULL}) on 5 v1.1.1-unchanged items")
    print("=" * 80)
    print("Same v1.1.1 production prompt (paraphrase recency clause active).")
    print(f"Prior gpt-4o-mini scores under v1.1.1: {PRIOR_GPT4O_MINI_V1_1_1}")
    print(f"Gold:                                    {GOLD}\n")

    results: list[dict] = []
    total_cost = 0.0
    for iid in ITEMS:
        item, output = _build_item_and_output(by_id[iid])
        score_result = await judge.score(item, output)
        prior = PRIOR_GPT4O_MINI_V1_1_1[iid]
        gold = GOLD[iid]
        score = score_result.score
        if isinstance(score, int) and score > prior:
            marker = f"→ GPT-4o disagrees with mini (mini={prior}, 4o={score})"
        elif score == prior:
            marker = f"= GPT-4o agrees with mini ({score})"
        else:
            marker = f"→ GPT-4o below mini ({score})"
        correctness = "✓ matches gold" if score == gold else f"✗ vs gold={gold}"
        print(f"  {iid}: 4o={score}  mini-prior={prior}  gold={gold}  {marker}  {correctness}")
        print(f"    reasoning: {score_result.reasoning[:300]}{'...' if len(score_result.reasoning) > 300 else ''}")
        print(f"    evidence_quotes: {score_result.evidence_quotes}")
        print()
        row = score_result.model_dump()
        row["item_id"] = iid
        row["mini_prior_score"] = prior
        row["gold_score"] = gold
        results.append(row)
        total_cost += score_result.cost_usd

    n_correct = sum(1 for r in results if r["score"] == r["gold_score"])
    n_disagree_with_mini = sum(
        1 for r in results
        if isinstance(r["score"], int) and r["score"] != r["mini_prior_score"]
    )
    print("=" * 80)
    print(f"GPT-4o correct (matches gold): {n_correct}/5")
    print(f"GPT-4o disagrees with gpt-4o-mini-v1.1.1: {n_disagree_with_mini}/5")
    print(f"Total cost: ${total_cost:.4f}")
    print()
    if n_correct >= 4:
        print("→ Residual is small-model-specific. v1.2 fix #3 (per-dim exclusion or")
        print("  stronger model on completeness) has clean empirical support.")
    elif n_correct >= 2:
        print("→ Mixed: GPT-4o handles some residuals but not all. Some failure modes")
        print("  are model-class limited; others may be rubric-limited.")
    else:
        print("→ Rubric is the limiting factor. Even GPT-4o struggles on these items")
        print("  with the v1.1.1 prompt. v1.2 needs rubric anchoring/simplification,")
        print("  not just judge-membership tuning.")

    out = REPO / "measurements/2026-05-06-4a-gpt4o-full-probe.jsonl"
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nProbe artifact: {out}")


if __name__ == "__main__":
    asyncio.run(main())
