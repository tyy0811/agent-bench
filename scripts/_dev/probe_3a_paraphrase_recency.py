"""Plan 3A probe: recency-positioned paraphrase instruction on 5 disputed
completeness items.

Sends the same rubric body, gold reference, and system answer as the
production CompletenessJudge prompt, with one extra sentence inserted
between the system answer and the score instruction:

  "Note: a paraphrase that captures the same meaning as a gold-answer
   point counts as covered. Score on content equivalence, not surface
   form."

Prior scores (from the v1 jury sidecar): all 5 disputed items scored 1
by gpt-4o-mini-2024-07-18; gold=2 on all 5; Haiku scored 2 on all 5.

Pre-committed criteria (DECISIONS "Plan 3A" entry):
  - Fixed:    ≥3/5 shift from 1 → 2
  - Partial:  1–2/5 shift
  - Not fix:  0/5 shift

Run:
    OPENAI_API_KEY=... python scripts/_dev/probe_3a_paraphrase_recency.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_bench.core.provider import OpenAIProvider  # noqa: E402
from agent_bench.core.types import Message, Role  # noqa: E402
from agent_bench.evaluation.judges.base import (  # noqa: E402
    Rubric,
    _strip_markdown_fence,
)

DISPUTED_IDS = ["q006", "q011", "k8s_002", "k8s_006", "k8s_018"]
PRIOR_SCORES = {iid: 1 for iid in DISPUTED_IDS}  # all five scored 1 in v1 sidecar
GOLD_SCORES = {iid: 2 for iid in DISPUTED_IDS}  # all five gold=2

PARAPHRASE_RECENCY_CLAUSE = (
    "Note: a paraphrase that captures the same meaning as a gold-answer "
    "point counts as covered. Score on content equivalence, not surface form."
)


def _load_outputs() -> dict[str, dict]:
    raw = (REPO / "results/calibration_v1_system_outputs.json").read_text()
    return {r["item_id"]: r for r in json.loads(raw)}


def _build_prompt(rubric: Rubric, item_record: dict) -> str:
    """Mirror CompletenessJudge.score's prompt construction, with the
    recency clause inserted between the system answer and the score
    instruction."""
    schema_clause = (
        'JSON object: {"reasoning": "...", "evidence_quotes": [...], '
        '"score": 0 or 1 or 2 or "Unknown"}.'
    )
    return (
        f"{rubric.render_prompt(level_permutation_seed=0)}\n\n"
        f"---\n\n"
        f"## Reference answer (gold)\n{item_record['reference_answer']}\n\n"
        f"## Answer to score\n{item_record['answer']}\n\n"
        f"{PARAPHRASE_RECENCY_CLAUSE}\n\n"
        f"Score this answer against the rubric above. Respond with ONLY a "
        f"{schema_clause}"
    )


def _parse_score(content: str) -> tuple[int | str, str, list[str]]:
    """Mirror _call_judge_with_retry's parse path: fence-strip then
    json.loads, return (score, reasoning, evidence_quotes)."""
    stripped = _strip_markdown_fence(content)
    data = json.loads(stripped)
    return (
        data["score"],
        str(data.get("reasoning", "")),
        list(data.get("evidence_quotes", [])),
    )


async def main() -> None:
    rubric = Rubric.from_markdown_file(
        REPO / "agent_bench/evaluation/rubrics/completeness.md"
    )
    outputs = _load_outputs()
    provider = OpenAIProvider(model="gpt-4o-mini-2024-07-18")

    print("=" * 80)
    print("Plan 3A — recency-positioned paraphrase instruction probe")
    print("=" * 80)
    print(f"prior scores: {PRIOR_SCORES}")
    print(f"gold scores:  {GOLD_SCORES}")
    print(f"intervention: \n  {PARAPHRASE_RECENCY_CLAUSE!r}\n")

    results: dict[str, dict] = {}
    total_cost = 0.0
    for iid in DISPUTED_IDS:
        item = outputs[iid]
        prompt = _build_prompt(rubric, item)
        response = await provider.complete(
            [Message(role=Role.USER, content=prompt)],
            temperature=0.0,
            max_tokens=1024,
        )
        try:
            score, reasoning, ev = _parse_score(response.content)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  {iid}: PARSE FAILED — {e}; raw={response.content[:200]!r}")
            continue
        prior = PRIOR_SCORES[iid]
        gold = GOLD_SCORES[iid]
        shifted = isinstance(score, int) and score > prior
        marker = "→ SHIFTED 1→2" if shifted else ("→ unchanged" if score == prior else f"→ shifted to {score}")
        print(f"  {iid}: prior={prior} new={score} gold={gold} {marker}")
        print(f"    reasoning: {reasoning[:300]}{'...' if len(reasoning) > 300 else ''}")
        print(f"    evidence_quotes: {ev}")
        print()
        results[iid] = {
            "prior": prior,
            "new": score,
            "gold": gold,
            "reasoning": reasoning,
            "evidence_quotes": ev,
            "shifted_up": shifted,
        }
        total_cost += response.usage.estimated_cost_usd

    n_shifted = sum(1 for r in results.values() if r["shifted_up"])
    print("=" * 80)
    print(f"Result: {n_shifted}/5 items shifted 1 → 2")
    print(f"Total cost: ${total_cost:.4f}")
    print()
    if n_shifted >= 3:
        print("→ FIXED (per pre-committed criteria). Re-run on full 26 disputed items.")
    elif n_shifted >= 1:
        print("→ PARTIALLY FIXED. Re-run on full 26 disputed items for clean number.")
    else:
        print("→ NOT FIXED. Escalate to 4A (GPT-4o full).")

    out_path = REPO / "measurements/2026-05-06-3a-paraphrase-recency-probe.jsonl"
    with out_path.open("w") as f:
        for iid, r in results.items():
            f.write(json.dumps({"item_id": iid, **r}) + "\n")
    print(f"\nProbe artifact: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
